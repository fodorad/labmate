"""Step 4 (HUMAN-IN-THE-LOOP): pause for the outline to be approved or edited.

The pipeline writes ``outline.yaml`` and stops. The human edits it (reorder, delete,
retitle, swap claim ids), then resumes with ``--approve``. The edited file is validated
with the same rules as the model's outline, so a human can't break the pipeline either.
This is the cheapest point to fix direction: every step after it costs more.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from paper2flow.schemas import Claims, Outline, PaperType
from paper2flow.steps.outline import check_outline, format_claims


class GateError(ValueError):
    """Raised when an edited outline is invalid."""


HEADER = """\
# paper2flow outline for review.
# Edit freely: change the titles, the hook or the claim ids (1-5 per slide).
# The four slides stay: {purposes}, in this order.
# Then continue with:  make approve ARXIV={paper_id}
"""


def write_gate(
    path: Path, outline: Outline, claims: Claims, paper_id: str, allowed: list[str]
) -> Path:
    """Write the outline as an editable YAML file, with the claim cards as a comment block.

    Args:
        path: Destination (``outline.yaml``).
        outline: Draft outline.
        claims: Claim cards (listed for reference).
        paper_id: Used in the resume instruction.
        allowed: Allowed purposes.

    Returns:
        ``path``.
    """
    body = yaml.safe_dump(outline.model_dump(), sort_keys=False, allow_unicode=True, width=100)
    reference = "\n".join(f"#   {line}" for line in format_claims(claims).splitlines())
    path.write_text(
        HEADER.format(purposes=", ".join(allowed), paper_id=paper_id)
        + "\n"
        + body
        + "\n# Available claim cards:\n"
        + reference
        + "\n"
    )
    return path


def read_gate(path: Path, claims: Claims, paper_type: PaperType) -> Outline:
    """Load and validate a (possibly human-edited) outline file.

    Args:
        path: ``outline.yaml``.
        claims: Claim cards the ids must refer to.
        paper_type: Routed paper type (selects the allowed purposes).

    Returns:
        The approved outline.

    Raises:
        GateError: If the file is missing, malformed or breaks the outline rules.
    """
    if not path.exists():
        raise GateError(f"{path} not found; run without --approve first")
    try:
        outline = Outline.model_validate(yaml.safe_load(path.read_text()))
    except (yaml.YAMLError, ValidationError) as e:
        raise GateError(f"{path} is not a valid outline:\n{e}") from e
    problems = check_outline(outline, claims, paper_type)
    if problems:
        raise GateError(f"{path} breaks the outline rules:\n- " + "\n- ".join(problems))
    return outline
