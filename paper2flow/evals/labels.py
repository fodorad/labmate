"""A blind labelling sheet: bullets and their evidence, for a human to judge.

The sheet samples bullets from every fact-check round of the given runs, stratified so
that bullets the pipeline's judge rejected are well represented (a sheet of only
supported bullets can't tell a good judge from one that always says "supported"). The
pipeline's verdict is kept out of the CSV, so the human label is independent.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import cast, get_args

from pydantic import BaseModel

from paper2flow.schemas import Claims, FactChecked, VerdictLabel

COLUMNS = ["id", "paper", "text", "evidence", "human"]
"""CSV columns, in order."""

EVIDENCE_SEP = " || "
"""Separator between evidence quotes in the CSV."""

ALIASES: dict[str, VerdictLabel] = {"s": "supported", "p": "partial", "u": "unsupported"}
"""Shorthands accepted in the ``human`` column."""


class LabelledBullet(BaseModel):
    """One bullet to label.

    Attributes:
        id: ``<paper>/r<round>s<slide>b<bullet>``.
        paper: Run directory name.
        text: Bullet text.
        evidence: The verified quotes the bullet cites.
        pipeline: The pipeline judge's verdict (not exported).
        human: Human label, empty until labelled.
    """

    id: str
    paper: str
    text: str
    evidence: list[str]
    pipeline: VerdictLabel | None = None
    human: VerdictLabel | None = None


def bullets_for_labelling(run_dir: Path) -> list[LabelledBullet]:
    """All distinct bullets the fact-check saw in a run, with their evidence.

    Args:
        run_dir: Run directory with ``02_claims.json`` and ``05_factcheck.json``.

    Returns:
        One entry per distinct bullet text, first occurrence wins.
    """
    claims = Claims.model_validate_json((run_dir / "02_claims.json").read_text())
    checked = FactChecked.model_validate_json((run_dir / "05_factcheck.json").read_text())
    quotes = {c.id: c.evidence_quote for c in claims.cards}
    seen: set[str] = set()
    out = []
    for r, checks in enumerate(checked.report.rounds):
        for c in checks:
            if c.text in seen:
                continue
            seen.add(c.text)
            out.append(
                LabelledBullet(
                    id=f"{run_dir.name}/r{r}s{c.slide}b{c.bullet}",
                    paper=run_dir.name,
                    text=c.text,
                    evidence=[quotes[i] for i in c.claim_ids if i in quotes],
                    pipeline=c.verdict,
                )
            )
    return out


def sample(bullets: list[LabelledBullet], n: int, seed: int = 0) -> list[LabelledBullet]:
    """Stratified sample: up to half rejected by the pipeline judge, the rest accepted.

    Args:
        bullets: Candidates.
        n: Sample size.
        seed: Random seed.

    Returns:
        At most ``n`` bullets in shuffled order.
    """
    rng = random.Random(seed)
    rejected = [b for b in bullets if b.pipeline != "supported"]
    accepted = [b for b in bullets if b.pipeline == "supported"]
    take_rejected = min(len(rejected), max(n // 2, n - len(accepted)))
    picked = rng.sample(rejected, take_rejected)
    picked += rng.sample(accepted, min(len(accepted), n - take_rejected))
    rng.shuffle(picked)
    return picked


def parse_label(value: str) -> VerdictLabel | None:
    """Parse a human label, accepting ``s`` / ``p`` / ``u`` shorthands.

    Args:
        value: Cell content.

    Returns:
        The label, or ``None`` for an empty cell.

    Raises:
        ValueError: For anything else.
    """
    value = value.strip().lower()
    if not value:
        return None
    if value in ALIASES:
        return ALIASES[value]
    if value in get_args(VerdictLabel):
        return cast(VerdictLabel, value)
    raise ValueError(f"unknown label {value!r}; use supported/partial/unsupported or s/p/u")


def read_labels(path: Path) -> list[LabelledBullet]:
    """Read a labelling sheet.

    Args:
        path: CSV written by :func:`export_labels`.

    Returns:
        All rows; ``human`` is ``None`` where not labelled yet.
    """
    with path.open(newline="") as f:
        return [
            LabelledBullet(
                id=row["id"],
                paper=row["paper"],
                text=row["text"],
                evidence=[e for e in row["evidence"].split(EVIDENCE_SEP) if e],
                human=parse_label(row["human"]),
            )
            for row in csv.DictReader(f)
        ]


def export_labels(run_dirs: list[Path], out: Path, n: int = 50, seed: int = 0) -> int:
    """Write (or top up) the labelling sheet.

    Existing rows and their labels are kept; new bullets are added until the sheet has
    ``n`` rows.

    Args:
        run_dirs: Finished runs to sample from.
        out: CSV path.
        n: Target number of rows.
        seed: Sampling seed.

    Returns:
        Number of rows added.
    """
    existing = read_labels(out) if out.exists() else []
    known = {b.id for b in existing} | {b.text for b in existing}
    candidates = [
        b
        for d in run_dirs
        for b in bullets_for_labelling(d)
        if b.id not in known and b.text not in known
    ]
    added = sample(candidates, max(0, n - len(existing)), seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for b in [*existing, *added]:
            writer.writerow(
                {
                    "id": b.id,
                    "paper": b.paper,
                    "text": b.text,
                    "evidence": EVIDENCE_SEP.join(b.evidence),
                    "human": b.human or "",
                }
            )
    return len(added)
