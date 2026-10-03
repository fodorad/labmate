"""The model steps of cv2job: read the posting, match it to the CV, tailor, write highlights.

Each step returns validated data and checks what the model says against the CV or the posting
in code, feeding problems back to the model: a quote must be in the posting, an id must be in
the CV, a reworded bullet may not bring a number or name its source lacks.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from labmate.core.extract import QUOTE_MATCH, quote_score
from labmate.core.factcheck import deterministic_problems, names_in
from labmate.core.lc import batch_map, prompt, structured
from labmate.core.schemas import ClaimCard
from labmate.cv2job.prompts import load_prompt
from labmate.cv2job.schemas import (
    Cv,
    Finding,
    Highlight,
    Highlights,
    Job,
    JobDraft,
    Match,
    MatchDraft,
    Requirement,
    Role,
    TailoredRole,
)

MAX_REQUIREMENTS = 12
"""Most requirements read from a posting."""

MAX_HIGHLIGHTS = 4
"""Most experiences the cover letter points to."""


_WORD = re.compile(r"[A-Za-z][A-Za-z0-9+#.\-]*")


def proper_nouns(text: str) -> set[str]:
    """Names of tools, products and methods in ``text``, lower-cased.

    A capitalised word in the middle of a line (``Kubernetes``, ``Docker``) or an identifier
    such as ``PyTorch``, ``SQL`` or ``F1``. The first word of a line is not counted.

    Args:
        text: Any text.

    Returns:
        The names.
    """
    names = set()
    for line in text.splitlines():
        for i, word in enumerate(_WORD.findall(line)):
            if (i > 0 and word[0].isupper()) or names_in(word):
                names.add(word.rstrip(".").lower())
    return names


def invented_names(text: str, source: str, posting: str) -> list[str]:
    """Names a reworded text brings that its source does not.

    This is what the shared number-and-identifier check misses for a CV: a plain tool name
    such as Kubernetes. It covers capitalised names in the text itself and any name from the
    posting that the text uses. Ordinary words of the posting may be used freely: wording a
    bullet in the posting's terms is the point of tailoring.

    Args:
        text: The reworded bullet or sentence.
        source: The CV text it must come from.
        posting: The job posting.

    Returns:
        Names the source does not contain.
    """
    words = {w.rstrip(".").lower() for w in _WORD.findall(text)}
    suspects = proper_nouns(text) | (proper_nouns(posting) & words)
    known = {w.rstrip(".").lower() for w in _WORD.findall(source)}
    return sorted(suspects - known)


def load_cv(path: Path) -> Cv:
    """Read ``cv.yaml``.

    Args:
        path: The YAML file.

    Returns:
        The CV, with an id on every bullet.
    """
    return Cv.model_validate(yaml.safe_load(path.read_text()))


def format_cv(cv: Cv) -> str:
    """The CV as the models see it: skills, then every bullet with its id and role.

    Args:
        cv: The CV.

    Returns:
        Text with one line per bullet (``[r1.2] Role, Company: text``).
    """
    skills = ", ".join(f"{s.name} (since {s.since})" if s.since else s.name for s in cv.skills)
    lines = [f"Skills: {skills or '(none)'}"]
    lines += [f"[{b.id}] {r.title}, {r.company}: {b.text}" for r in cv.roles for b in r.bullets]
    return "\n".join(lines)


def read_job(job_text: str, model: BaseChatModel, config: RunnableConfig | None = None) -> Job:
    """Read the requirements of a posting; each quote must be in the posting.

    Args:
        job_text: The posting.
        model: The writer model.
        config: The calling step's config (callbacks).

    Returns:
        The job, requirements numbered ``q1``, ``q2``, ...
    """

    def check(draft: JobDraft) -> list[str]:
        problems = []
        if not 1 <= len(draft.requirements) <= MAX_REQUIREMENTS:
            problems.append(f"list 1 to {MAX_REQUIREMENTS} requirements")
        for i, req in enumerate(draft.requirements, start=1):
            if quote_score(req.quote, job_text) < QUOTE_MATCH:
                problems.append(
                    f"requirement {i}: the quote is not in the posting; copy it exactly"
                )
        return problems

    chain = prompt(load_prompt("requirements")) | structured(model, JobDraft, check=check)
    draft = chain.invoke({"job": job_text}, config)
    return Job(
        position=draft.position,
        company=draft.company,
        requirements=[
            Requirement(id=f"q{i}", **req.model_dump())
            for i, req in enumerate(draft.requirements, start=1)
        ],
    )


def match_requirements(
    cv: Cv,
    job: Job,
    model: BaseChatModel,
    workers: int = 2,
    config: RunnableConfig | None = None,
) -> list[Match]:
    """Find CV evidence for every requirement (one call each, in parallel).

    Args:
        cv: The CV.
        job: The posting's requirements.
        model: The writer model.
        workers: Concurrent calls.
        config: The calling step's config (callbacks).

    Returns:
        One match per requirement; empty evidence means the CV shows nothing.
    """
    known_bullets, known_skills = set(cv.bullets()), {s.name for s in cv.skills}
    text = format_cv(cv)

    def check(draft: MatchDraft) -> list[str]:
        problems = []
        if unknown := sorted(set(draft.bullet_ids) - known_bullets):
            problems.append(f"unknown bullet ids {unknown}")
        if unknown := sorted(set(draft.skills) - known_skills):
            problems.append(f"unknown skills {unknown}")
        return problems

    chain = prompt(load_prompt("match")) | structured(model, MatchDraft, check=check)

    def match(req: Requirement, cfg: RunnableConfig) -> Match:
        draft = chain.invoke({"requirement": req.text, "quote": req.quote, "cv": text}, cfg)
        return Match(requirement_id=req.id, **draft.model_dump())

    return batch_map(match, job.requirements, config, workers)


def as_evidence(cv: Cv, bullet_ids: list[str], fact: str = "") -> list[ClaimCard]:
    """CV bullets (and what the candidate said) as claim cards, for the shared checks.

    Args:
        cv: The CV.
        bullet_ids: Bullets to include.
        fact: What the candidate said, if it is evidence.

    Returns:
        One card per piece of evidence, quote = its text.
    """
    bullets = cv.bullets()
    cards = [
        ClaimCard(id=i, claim=b.text, evidence_quote=b.text, kind="result", section=r.title, page=0)
        for i in bullet_ids
        for r, b in [bullets[i]]
    ]
    if fact:
        cards.append(
            ClaimCard(
                id="answer", claim=fact, evidence_quote=fact, kind="result", section="", page=0
            )
        )
    return cards


def tailor_roles(
    cv: Cv,
    job: Job,
    model: BaseChatModel,
    workers: int = 2,
    config: RunnableConfig | None = None,
) -> list[TailoredRole]:
    """Choose, order and lightly reword each role's bullets for the job.

    A reworded bullet may not carry a number or a name its source bullet lacks.

    Args:
        cv: The CV.
        job: The posting's requirements.
        model: The writer model.
        workers: Concurrent calls (one per role).
        config: The calling step's config (callbacks).

    Returns:
        One tailored role per CV role, in CV order.
    """
    wanted = "\n".join(f"- {r.text}" + (" (must)" if r.must else "") for r in job.requirements)
    posting = "\n".join(r.quote for r in job.requirements)

    def tailor(role: Role, cfg: RunnableConfig) -> TailoredRole:
        own = {b.id: b for b in role.bullets}

        def check(draft: TailoredRole) -> list[str]:
            problems = []
            ids = [b.source_id for b in draft.bullets]
            if unknown := sorted(set(ids) - set(own)):
                problems.append(f"source ids {unknown} are not bullets of this role")
            if len(set(ids)) != len(ids):
                problems.append("each source bullet may be used once")
            for b in draft.bullets:
                if b.source_id in own:
                    problems += [
                        f"bullet from {b.source_id}: {p}"
                        for p in deterministic_problems(b.text, as_evidence(cv, [b.source_id]))
                    ]
                    if names := invented_names(b.text, own[b.source_id].text, posting):
                        problems.append(
                            f"bullet from {b.source_id}: {', '.join(names)} "
                            "is not named in the source bullet"
                        )
            return problems

        chain = prompt(load_prompt("tailor")) | structured(model, TailoredRole, check=check)
        out = chain.invoke(
            {
                "position": job.position,
                "requirements": wanted,
                "title": role.title,
                "company": role.company,
                "bullets": "\n".join(f"[{b.id}] {b.text}" for b in role.bullets),
            },
            cfg,
        )
        return out.model_copy(update={"role_id": role.id})

    return batch_map(tailor, cv.roles, config, workers)


def covered_requirements(
    job: Job, matches: list[Match], findings: list[Finding]
) -> list[tuple[Requirement, list[str], str]]:
    """The requirements the CV or the candidate shows, must-haves first.

    Args:
        job: The posting's requirements.
        matches: CV evidence per requirement.
        findings: The gap agent's conclusions.

    Returns:
        ``(requirement, bullet ids, candidate's fact)``; skill-only matches have no bullets.
    """
    found = {f.requirement_id: f for f in findings if f.status == "covered"}
    out = []
    for req in job.requirements:
        match = next((m for m in matches if m.requirement_id == req.id), None)
        if match and match.covered:
            out.append((req, match.bullet_ids, ""))
        elif req.id in found:
            out.append((req, found[req.id].bullet_ids, found[req.id].fact))
    return sorted(out, key=lambda t: not t[0].must)


def write_highlights(
    cv: Cv,
    job: Job,
    matches: list[Match],
    findings: list[Finding],
    model: BaseChatModel,
    config: RunnableConfig | None = None,
) -> list[Highlight]:
    """The experiences the cover letter points to, one reason each.

    Args:
        cv: The CV.
        job: The posting's requirements.
        matches: CV evidence per requirement.
        findings: The gap agent's conclusions.
        model: The writer model.
        config: The calling step's config (callbacks).

    Returns:
        Up to :data:`MAX_HIGHLIGHTS` highlights, must-haves first; none if nothing is covered.
    """
    chosen = covered_requirements(job, matches, findings)[:MAX_HIGHLIGHTS]
    if not chosen:
        return []
    bullets = cv.bullets()
    posting = "\n".join(r.quote for r in job.requirements)
    experiences, cards, source = [], {}, {}
    for req, ids, fact in chosen:
        evidence = [bullets[i][1].text for i in ids] + ([fact] if fact else [])
        match = next((m for m in matches if m.requirement_id == req.id), None)
        evidence += [f"skill: {s}" for s in (match.skills if match else [])]
        source[req.id] = " ".join(evidence)
        experiences.append(f"{req.id} ({req.text}):\n" + "\n".join(f"  - {e}" for e in evidence))
        cards[req.id] = as_evidence(cv, ids, fact) + [
            ClaimCard(id="skill", claim=e, evidence_quote=e, kind="result", section="", page=0)
            for e in evidence
            if e.startswith("skill: ")
        ]

    def check(h: Highlights) -> list[str]:
        problems = []
        for item in h.items:
            if item.requirement_id not in cards:
                problems.append(f"unknown requirement id {item.requirement_id}")
            else:
                problems += [
                    f"{item.requirement_id}: {p}"
                    for p in deterministic_problems(item.reason, cards[item.requirement_id])
                ]
                if names := invented_names(item.reason, source[item.requirement_id], posting):
                    problems.append(
                        f"{item.requirement_id}: {', '.join(names)} is not in its evidence"
                    )
        if len(h.items) != len(chosen):
            problems.append(f"write exactly {len(chosen)} highlights, one per experience")
        return problems

    chain = prompt(load_prompt("letter")) | structured(model, Highlights, check=check)
    return chain.invoke(
        {"position": job.position, "experiences": "\n\n".join(experiences)}, config
    ).items
