import json

import pymupdf

from labmate.config import Config, ReplayMode
from labmate.core.tracing import read_trace
from labmate.paper2flow.chain import paper2flow
from labmate.paper2post.chain import POST_ARTIFACT, paper2post
from tests.conftest import agentic_chat

REF = "2401.00001"


def config_in(tmp_path):
    cfg = Config()
    cfg.replay.dir = tmp_path / "cassettes"
    cfg.replay.lock_file = tmp_path / "models.lock"
    cfg.tracing.runs_dir = tmp_path / "runs"
    cfg.post.links = {"Website": "https://adamfodor.com"}
    return cfg


def test_a_paper_becomes_a_post_pdf(fake, arxiv, tmp_path):
    fake.chat_handler = agentic_chat
    post = paper2post(config_in(tmp_path), REF, client=fake.client(), transport=arxiv.transport())
    with pymupdf.open(post) as doc:
        assert doc.page_count == 2  # the text, then the pipeline image
        text = " ".join(doc[0].get_text().split())
        assert "Linear attention without the accuracy tax" in text
        assert "Website: https://adamfodor.com" in text
        assert doc[1].get_images()
    saved = json.loads((post.parent / POST_ARTIFACT).read_text())
    assert len(saved["icons"]) == len(saved["takeaways"]) > 0
    steps = {s["name"] for s in read_trace(post.parent / "trace.jsonl")}
    assert {"step.flows", "step.post", "step.icons", "step.render"} <= steps
    assert not fake.loaded


def test_the_post_reuses_the_overview_analysis(fake, arxiv, tmp_path):
    config = config_in(tmp_path)
    fake.chat_handler = agentic_chat
    paper2flow(config, REF, client=fake.client(), transport=arxiv.transport())
    before = fake.paths().count("/api/chat")
    paper2post(config, REF, client=fake.client(), transport=arxiv.transport())
    # only the post's own calls are new: draft, judge, icons
    assert fake.paths().count("/api/chat") - before == 3
    again = paper2post(config, REF, mode=ReplayMode.REPLAY)
    assert again.exists()
