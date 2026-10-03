"""The cover letter: a fixed, plain template. Code writes everything but the highlight reasons."""

from __future__ import annotations

from datetime import date

from labmate.cv2job.schemas import Cv, Highlight, Job

INTRO = "These are the experiences from my CV that fit the role best:"
CLOSING = "Thank you for reading my application. I look forward to hearing from you."


def years_sentence(cv: Cv, highlights: list[Highlight], today: date) -> str:
    """How long the candidate has used the skill the highlights lean on most.

    The years come from the CV's ``since`` year, never from the model.

    Args:
        cv: The CV.
        highlights: The letter's highlights.
        today: The current date.

    Returns:
        ``I have N years of experience in <skill>.``, or ``""`` if no CV skill with a start
        year appears in the highlights.
    """
    text = " ".join(f"{h.skill} {h.reason}" for h in highlights).lower()
    used = [s for s in cv.skills if s.since and s.name.lower() in text]
    if not used:
        return ""
    skill = min(used, key=lambda s: s.since or today.year)
    years = max(today.year - (skill.since or today.year), 1)
    return f"I have {years} {'year' if years == 1 else 'years'} of experience in {skill.name}."


def letter_data(cv: Cv, job: Job, highlights: list[Highlight], today: date) -> dict[str, object]:
    """The letter as the template prints it.

    Args:
        cv: The CV.
        job: The posting (position and company).
        highlights: Up to four experiences with their reasons.
        today: The current date (for the years of experience).

    Returns:
        The opening paragraph, the highlights, the closing and the signature.
    """
    company = job.company or "your company"
    opening = (
        f"I am applying for the {job.position} position at {company}. I compared the "
        "responsibilities with my skills and interests, and I decided to apply."
    )
    if years := years_sentence(cv, highlights, today):
        opening += f" {years}"
    return {
        "name": cv.name,
        "opening": opening,
        "intro": INTRO if highlights else "",
        "highlights": [{"skill": h.skill, "reason": h.reason} for h in highlights],
        "closing": CLOSING,
    }
