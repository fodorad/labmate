"""The gap agent: the one step of cv2job where the model decides what to do.

Requirements that matched nothing in the CV are handed to a tool-calling agent. It decides how
to look (it can search the CV with other words), whether to ask the candidate, and when it is
done. Its conclusions are checked by the tools themselves: a ``covered`` finding needs CV
bullets that exist, or the candidate's own answer, which is stored verbatim, never as the model
wrote it. A requirement the agent never reports stays a gap.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from labmate.cv2job.prompts import load_prompt
from labmate.cv2job.schemas import Cv, Finding, Job, Match

RECURSION_LIMIT = 40
"""Graph steps the agent may take (each tool call is two)."""

Ask = Callable[[str], str]
"""Asks the candidate a question and returns the answer ('' if there is none)."""


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9+#.]{3,}", text.lower())}


def search_cv(cv: Cv, query: str, limit: int = 5) -> str:
    """The CV bullets and skills that share the most words with ``query``.

    Args:
        cv: The CV.
        query: What to look for.
        limit: Most results.

    Returns:
        One line per hit (``[id] role: text``), or ``no results``.
    """
    wanted = _words(query)
    scored = [
        (len(wanted & _words(f"{r.title} {b.text}")), f"[{b.id}] {r.title}: {b.text}")
        for r in cv.roles
        for b in r.bullets
    ]
    scored += [(len(wanted & _words(s.name)), f"skill: {s.name}") for s in cv.skills]
    hits = [line for score, line in sorted(scored, key=lambda t: -t[0]) if score > 0][:limit]
    return "\n".join(hits) or "no results"


def find_gaps(
    cv: Cv,
    job: Job,
    matches: list[Match],
    model: BaseChatModel,
    ask: Ask,
    config: RunnableConfig | None = None,
) -> list[Finding]:
    """Run the agent on the requirements nothing matched.

    Args:
        cv: The CV.
        job: The posting's requirements.
        matches: CV evidence per requirement.
        model: The writer model (it must support tool calling).
        ask: Asks the candidate a question.
        config: The calling step's config (callbacks).

    Returns:
        One finding per requirement without a match; none if every requirement matched.
    """
    covered = {m.requirement_id for m in matches if m.covered}
    open_reqs = {r.id: r for r in job.requirements if r.id not in covered}
    if not open_reqs:
        return []
    bullets = cv.bullets()
    findings: dict[str, Finding] = {}
    answers: list[str] = []

    @tool("search_cv")
    def search(query: str) -> str:
        """Search the CV's bullets and skills.

        Args:
            query: What to look for.
        """
        return search_cv(cv, query)

    @tool("read_cv_item")
    def read(item_id: str) -> str:
        """Read one CV bullet with its role.

        Args:
            item_id: A bullet id such as r1.2.
        """
        if item_id not in bullets:
            return f"unknown id {item_id!r}"
        role, bullet = bullets[item_id]
        return f"[{item_id}] {role.title}, {role.company}: {bullet.text}"

    @tool("ask_candidate")
    def ask_candidate(question: str) -> str:
        """Ask the candidate one short question.

        Args:
            question: The question.
        """
        answer = ask(question).strip()
        if answer:
            answers.append(answer)
        return answer or "(the candidate gave no answer)"

    @tool("report_finding")
    def report_finding(
        requirement_id: str,
        status: Literal["covered", "gap"],
        bullet_ids: list[str] | None = None,
        from_answer: bool = False,
    ) -> str:
        """Record the conclusion for one requirement.

        Args:
            requirement_id: The requirement, e.g. q3.
            status: covered if the CV or the candidate's answer shows it, else gap.
            bullet_ids: CV bullets that show it.
            from_answer: True if the candidate's last answer shows it.
        """
        ids = bullet_ids or []
        if requirement_id not in open_reqs:
            return f"unknown requirement {requirement_id!r}; use one of {sorted(open_reqs)}"
        if status == "covered":
            if unknown := [i for i in ids if i not in bullets]:
                return f"unknown bullet ids {unknown}"
            if not ids and not (from_answer and answers):
                return "covered needs CV bullet ids, or from_answer after the candidate answered"
        fact = answers[-1] if status == "covered" and from_answer and answers else ""
        findings[requirement_id] = Finding(
            requirement_id=requirement_id, status=status, bullet_ids=ids, fact=fact
        )
        return "recorded"

    tools: list[BaseTool] = [search, read, ask_candidate, report_finding]
    listing = "\n".join(f'{r.id}: {r.text} (posting: "{r.quote}")' for r in open_reqs.values())
    agent = create_agent(
        model=model, tools=tools, system_prompt=load_prompt("gaps").format(requirements=listing)
    )
    agent.invoke(
        {"messages": [HumanMessage("Check the requirements without a match.")]},
        {**(config or {}), "recursion_limit": RECURSION_LIMIT},
    )
    return [findings.get(rid, Finding(requirement_id=rid, status="gap")) for rid in open_reqs]
