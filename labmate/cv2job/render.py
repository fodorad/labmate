"""The three PDFs of cv2job, laid out with Typst. Plain code, no model."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from labmate.core.theme import Theme
from labmate.cv2job.letter import letter_data
from labmate.cv2job.schemas import Application, Cv, Match
from labmate.cv2job.steps import covered_requirements
from labmate.paper2flow.chain import need
from labmate.paper2flow.steps.render import compile_typst

PACKAGE = "labmate.cv2job.templates"


def order_skills(cv: Cv, matches: list[Match]) -> list[str]:
    """The CV's skills, those the posting's requirements matched first.

    Args:
        cv: The CV.
        matches: CV evidence per requirement.

    Returns:
        Skill names.
    """
    matched = {s for m in matches for s in m.skills}
    names = [s.name for s in cv.skills]
    return sorted(names, key=lambda n: n not in matched)


def cv_data(app: Application) -> dict[str, Any]:
    """The tailored CV as the template prints it.

    Args:
        app: The finished application.

    Returns:
        Header, ordered skills, roles with their tailored bullets, education.
    """
    cv = app.cv
    tailored = {t.role_id: t for t in app.tailored}
    return {
        "name": cv.name,
        "headline": cv.headline,
        "contact": " · ".join(x for x in (cv.email, cv.location, *cv.links) if x),
        "skills": order_skills(cv, app.matches),
        "roles": [
            {
                "title": r.title,
                "company": r.company,
                "period": f"{r.start} – {r.end}",
                "bullets": [b.text for b in tailored[r.id].bullets],
            }
            for r in cv.roles
            if r.id in tailored
        ],
        "education": [e.model_dump() for e in cv.education],
    }


def gap_data(app: Application) -> dict[str, Any]:
    """The gap report as the template prints it.

    Args:
        app: The finished application (with a job).

    Returns:
        What is covered (and by what), what is not, and what the candidate said.
    """
    job = need(app.job, "job")
    bullets = app.cv.bullets()
    covered_rows = covered_requirements(job, app.matches, app.findings)
    covered = []
    for req, ids, fact in covered_rows:
        match = next((m for m in app.matches if m.requirement_id == req.id), None)
        evidence = [bullets[i][1].text for i in ids] + [
            f"skill: {s}" for s in (match.skills if match else [])
        ]
        covered.append(
            {
                "requirement": req.text,
                "must": req.must,
                "source": "from you" if fact and not ids else "your CV",
                "evidence": evidence + ([f"you said: {fact}"] if fact else []),
            }
        )
    done = {req.id for req, _, _ in covered_rows}
    return {
        "position": job.position,
        "company": job.company,
        "covered": covered,
        "gaps": [
            {"requirement": r.text, "must": r.must, "quote": r.quote}
            for r in job.requirements
            if r.id not in done
        ],
        "told": [f.fact for f in app.findings if f.fact],
    }


def render_all(app: Application, out_dir: Path, today: date) -> list[Path]:
    """Write ``cv.pdf``, ``cover_letter.pdf`` and ``gap_report.pdf``.

    Args:
        app: The finished application.
        out_dir: Output directory.
        today: The current date (for the letter's years of experience).

    Returns:
        The three PDF paths.
    """
    job = need(app.job, "job")
    theme = Theme().model_dump()
    pages = [
        ("cv.typ", "cv.pdf", cv_data(app)),
        ("letter.typ", "cover_letter.pdf", letter_data(app.cv, job, app.highlights, today)),
        ("gap_report.typ", "gap_report.pdf", gap_data(app)),
    ]
    return [
        compile_typst(PACKAGE, template, {**data, "theme": theme}, out_dir / pdf)
        for template, pdf, data in pages
    ]
