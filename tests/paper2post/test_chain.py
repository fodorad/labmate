import json

import pymupdf

from labmate.config import Config
from labmate.paper2flow.chain import paper2flow
from labmate.paper2post.chain import POST_ARTIFACT, paper2post
from tests.conftest import agentic_chat

REF = "2401.00001"


def config_in(tmp_path):
    cfg = Config()
    cfg.cache.path = tmp_path / "cache" / "replies.sqlite"
    cfg.tracing.runs_dir = tmp_path / "runs"
    cfg.post.links = {"Website": "https://adamfodor.com"}
    return cfg


def test_a_paper_becomes_a_post_pdf(fake, arxiv, tmp_path):
    fake.chat_handler = agentic_chat
    post = paper2post(config_in(tmp_path), REF, web=arxiv.transport(), ollama=fake.transport())
    with pymupdf.open(post) as doc:
        assert doc.page_count == 2  # the text, then the pipeline image
        text = " ".join(doc[0].get_text().split())
        assert "Linear attention without the accuracy tax" in text
        assert "Website: https://adamfodor.com" in text
        assert doc[1].get_images()
    saved = json.loads((post.parent / POST_ARTIFACT).read_text())
    assert len(saved["icons"]) == len(saved["takeaways"]) > 0


def test_the_post_reuses_the_overview_analysis(fake, arxiv, tmp_path):
    config = config_in(tmp_path)
    fake.chat_handler = agentic_chat
    paper2flow(config, REF, web=arxiv.transport(), ollama=fake.transport())
    before = fake.paths().count("/api/chat")
    paper2post(config, REF, web=arxiv.transport(), ollama=fake.transport())
    # only the post's own calls are new: draft, judge
    assert fake.paths().count("/api/chat") - before == 2
    done = fake.paths().count("/api/chat")
    paper2post(config, REF, web=arxiv.transport(), ollama=fake.transport())
    assert fake.paths().count("/api/chat") == done  # a rerun is answered from the cache
