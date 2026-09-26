"""LinkedIn post (extra output): drafted from the final slides, fact-checked like them.

The takeaways are treated as one slide and go through the same evaluator-optimizer loop,
so the post can't state anything the carousel couldn't.
"""

from __future__ import annotations

from paper2carousel.llm.structured import structured_chat
from paper2carousel.phases import ModelSwitcher
from paper2carousel.schemas import (
    Claims,
    Paper,
    Post,
    PostDraft,
    SlideText,
    WrittenSlides,
)
from paper2carousel.steps.factcheck import fact_check
from paper2carousel.steps.llm import LLM, load_prompt


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
        return [
            f"takeaway {i} cites unknown claims {sorted(set(t.claim_ids) - known)}"
            for i, t in enumerate(d.takeaways, start=1)
            if set(t.claim_ids) - known
        ]

    draft = structured_chat(writer.backend, writer.request(prompt), PostDraft, check=check)
    as_slide = WrittenSlides(
        hook=draft.hook, slides=[SlideText(title=draft.hook, bullets=draft.takeaways)]
    )
    checked = fact_check(as_slide, [used], claims, writer, judge, switcher, max_rounds)
    takeaways = checked.slides.slides[0].bullets if checked.slides.slides else []
    return Post(
        hook=draft.hook, takeaways=takeaways, question=draft.question, report=checked.report
    )


def post_markdown(post: Post, paper: Paper) -> str:
    """Render the post as paste-ready text.

    Args:
        post: Fact-checked post.
        paper: The paper (title and link).

    Returns:
        Markdown/plain text for LinkedIn.
    """
    lines = [post.hook, ""]
    lines += [f"→ {t.text}" for t in post.takeaways]
    lines += ["", post.question, "", f"Paper: {paper.title}"]
    if paper.url:
        lines.append(paper.url)
    lines += [
        "",
        "Made with paper2carousel: every claim on the slides is traced to a quote in the paper.",
    ]
    return "\n".join(lines) + "\n"
