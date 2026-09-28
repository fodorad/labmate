"""LinkedIn post text: drafted from the four fact-checked blocks, fact-checked like them.

The post body (3 to 5 sentences telling the paper's story) goes through the same
evaluator-optimizer loop as the blocks, so the post can't state anything the overview
couldn't; a sentence that stays unsupported is dropped.
"""

from __future__ import annotations

from labmate.core.factcheck import INLINE_ID, fact_check
from labmate.core.llm.structured import structured_chat
from labmate.core.model import LLM
from labmate.core.phases import ModelSwitcher
from labmate.paper2flow.prompts import load_prompt
from labmate.paper2flow.schemas import (
    Claims,
    Paper,
    Post,
    PostDraft,
    SlideText,
    WrittenSlides,
)


def _format_slides(slides: WrittenSlides) -> str:
    return "\n".join(
        f"- {b.text} [{', '.join(b.claim_ids)}]" for s in slides.slides for b in s.bullets
    )


def write_post(
    slides: WrittenSlides,
    claims: Claims,
    paper: Paper,
    writer: LLM,
    judge: LLM,
    switcher: ModelSwitcher,
    max_rounds: int = 1,
) -> Post:
    """Draft and fact-check the post.

    Args:
        slides: Final, fact-checked slides.
        claims: Claim cards.
        paper: The paper (title for the prompt).
        writer: Writer model settings.
        judge: Critic model settings.
        switcher: Model phases.
        max_rounds: Rewrite rounds for the takeaways.

    Returns:
        The post with only supported takeaways, and the audit trail.
    """
    used = sorted({cid for s in slides.slides for b in s.bullets for cid in b.claim_ids})
    switcher.use(writer.model)
    prompt = load_prompt("post").format(title=paper.title, slides=_format_slides(slides))
    known = set(used)

    def check(d: PostDraft) -> list[str]:
        problems = [
            f"takeaway {i} cites unknown claims {sorted(set(t.claim_ids) - known)}"
            for i, t in enumerate(d.takeaways, start=1)
            if set(t.claim_ids) - known
        ]
        return problems + [
            f"takeaway {i} has claim ids in its text; list them in claim_ids only"
            for i, t in enumerate(d.takeaways, start=1)
            if INLINE_ID.search(t.text)
        ]

    draft = structured_chat(writer.backend, writer.request(prompt), PostDraft, check=check)
    # The fact-check works on slides of at most three bullets, so the sentences are
    # checked in groups of three and put back together in order.
    groups = [draft.takeaways[i : i + 3] for i in range(0, len(draft.takeaways), 3)]
    as_slides = WrittenSlides(
        hook=draft.hook, slides=[SlideText(title=draft.hook, bullets=g) for g in groups]
    )
    checked = fact_check(
        as_slides,
        [used] * len(groups),
        claims,
        writer,
        judge,
        switcher,
        max_rounds,
        paper_title=paper.title,
    )
    takeaways = [b for slide in checked.slides.slides for b in slide.bullets]
    return Post(
        hook=draft.hook, takeaways=takeaways, question=draft.question, report=checked.report
    )
