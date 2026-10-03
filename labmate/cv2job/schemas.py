"""cv2job data model: the CV, the job, and what each step finds out."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Bullet(BaseModel):
    """One achievement of a role. Its id is what every later step cites."""

    id: str = ""
    text: str


class Role(BaseModel):
    """A position held."""

    id: str
    title: str
    company: str
    start: str
    end: str = "present"
    bullets: list[Bullet]

    @model_validator(mode="after")
    def _number_the_bullets(self) -> Role:
        for i, bullet in enumerate(self.bullets, start=1):
            bullet.id = bullet.id or f"{self.id}.{i}"
        return self


class Skill(BaseModel):
    """A skill and, if given, the year the candidate started using it."""

    name: str
    since: int | None = None


class Education(BaseModel):
    """A degree."""

    degree: str
    school: str
    year: str = ""


class Cv(BaseModel):
    """The candidate's CV, as written in ``cv.yaml``."""

    name: str
    headline: str = ""
    email: str = ""
    location: str = ""
    links: list[str] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    roles: list[Role]
    education: list[Education] = Field(default_factory=list)

    def bullets(self) -> dict[str, tuple[Role, Bullet]]:
        """Every bullet by id, with its role."""
        return {b.id: (r, b) for r in self.roles for b in r.bullets}


class RequirementDraft(BaseModel):
    """A requirement as the model reads it from the posting."""

    text: str = Field(description="The requirement in a few words, e.g. 'PyTorch experience'.")
    quote: str = Field(description="The sentence of the posting it comes from, copied exactly.")
    must: bool = Field(description="True for a must-have, false for a nice-to-have.")


class JobDraft(BaseModel):
    """The model's reading of a job posting."""

    position: str = Field(description="The job title as written in the posting.")
    company: str = Field(description="The employer's name as written, or '' if not named.")
    requirements: list[RequirementDraft]


class Requirement(RequirementDraft):
    """A requirement with its id."""

    id: str


class Job(BaseModel):
    """What the posting asks for."""

    position: str
    company: str = ""
    requirements: list[Requirement]


class MatchDraft(BaseModel):
    """CV evidence the model finds for one requirement."""

    bullet_ids: list[str] = Field(
        default_factory=list, description="Ids of the CV bullets that show it; empty if none."
    )
    skills: list[str] = Field(
        default_factory=list, description="Names of CV skills that show it; empty if none."
    )


class Match(MatchDraft):
    """Evidence for one requirement."""

    requirement_id: str

    @property
    def covered(self) -> bool:
        """Whether any CV evidence was found."""
        return bool(self.bullet_ids or self.skills)


class Finding(BaseModel):
    """What the gap agent concluded about a requirement nothing matched.

    Attributes:
        requirement_id: The requirement.
        status: ``covered`` if the CV or the candidate shows it, else ``gap``.
        bullet_ids: CV bullets that show it.
        fact: What the candidate said (verbatim), when that is the evidence; it is not in the CV.
    """

    requirement_id: str
    status: Literal["covered", "gap"]
    bullet_ids: list[str] = Field(default_factory=list)
    fact: str = ""


class TailoredBullet(BaseModel):
    """A CV bullet, reworded for the job, and the CV bullet it comes from."""

    source_id: str
    text: str


class TailoredRole(BaseModel):
    """A role with the bullets chosen and reordered for the job."""

    role_id: str
    bullets: list[TailoredBullet] = Field(min_length=1, max_length=4)


class Highlight(BaseModel):
    """One experience the cover letter points to."""

    requirement_id: str
    skill: str = Field(description="The skill or experience, at most 4 words.")
    reason: str = Field(description="One sentence, at most 25 words, from the evidence.")


class Highlights(BaseModel):
    """The model's highlights for the cover letter."""

    items: list[Highlight]


class Application(BaseModel):
    """What the chain knows so far; each step fills in one field.

    Attributes:
        cv: The candidate's CV.
        job_text: The posting as given.
        job: Its requirements.
        matches: CV evidence per requirement.
        findings: What the gap agent concluded for requirements without a match.
        tailored: The CV's roles, bullets chosen and reworded for the job.
        highlights: The experiences the cover letter points to.
    """

    cv: Cv
    job_text: str
    job: Job | None = None
    matches: list[Match] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    tailored: list[TailoredRole] = Field(default_factory=list)
    highlights: list[Highlight] = Field(default_factory=list)
