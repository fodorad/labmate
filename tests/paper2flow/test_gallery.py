import json
import shutil

import pytest

from labmate.config import Config
from labmate.paper2flow.engines.plain import run
from labmate.paper2flow.gallery import (
    Meta,
    PublishError,
    build_site,
    paper_context,
    publish,
    verify,
)
from tests.conftest import agentic_chat, make_pdf


@pytest.fixture
def config(tmp_path):
    cfg = Config()
    cfg.replay.dir = tmp_path / "cassettes"
    cfg.replay.lock_file = tmp_path / "models.lock"
    cfg.tracing.runs_dir = tmp_path / "runs"
    return cfg


@pytest.fixture
def finished(fake, arxiv, config):
    fake.chat_handler = agentic_chat
    run(config, ref="2401.00001", client=fake.client(), http=arxiv.client())
    return run(config, ref="2401.00001", client=fake.client(), http=arxiv.client(), approve=True)


@pytest.fixture
def entry(finished, config, arxiv, tmp_path):
    return publish(finished.run_dir, tmp_path / "gallery", config, http=arxiv.client())[0]


def test_publish_copies_artifacts_trace_and_only_the_runs_cassettes(
    finished, config, arxiv, tmp_path
):
    stray = config.replay.dir / "zz" / ("zz" + "0" * 62 + ".json")
    stray.parent.mkdir(parents=True)
    stray.write_text("{}")
    entry, n = publish(finished.run_dir, tmp_path / "gallery", config, http=arxiv.client())
    assert entry == tmp_path / "gallery" / "2401.00001"
    assert n == len(list((entry / "cassettes").glob("*/*.json"))) > 10
    assert not (entry / "cassettes" / "zz").exists()
    assert (entry / "05_factcheck.json").exists() and (entry / "09_flows.json").exists()
    assert (entry / "overview.pdf").exists() and (entry / "post.pdf").exists()
    assert not (entry / "flow.png").exists()  # the diagrams are inside the PDFs
    assert not (entry / "paper.pdf").exists() and not (entry / "00_paper.json").exists()
    meta = Meta.model_validate_json((entry / "meta.json").read_text())
    assert meta.source == "arxiv" and meta.title == "A Test Paper" and meta.published
    assert meta.publication.startswith("arXiv preprint")
    roots = [json.loads(line) for line in (entry / "trace.jsonl").read_text().splitlines()]
    assert {s["status"] for s in roots if s["name"] == "run"} == {"ok", "awaiting_approval"}
    # republishing replaces the entry
    (entry / "old.txt").write_text("x")
    publish(finished.run_dir, tmp_path / "gallery", config, http=arxiv.client())
    assert not (entry / "old.txt").exists()


def test_publish_finds_cassettes_of_traces_without_digest_keys(finished, config, arxiv, tmp_path):
    from labmate.core.llm.replay import CassetteStore

    config.replay.lock_file.write_text(json.dumps({"gemma4:26b-mlx": "cd" * 32}))
    store, legacy = CassetteStore(config.replay.dir), CassetteStore(tmp_path / "legacy")
    for path in store.root.glob("*/*.json"):  # re-key every cassette with the digest
        record = json.loads(path.read_text())
        from labmate.core.llm.types import ChatRequest

        if "messages" in record["request"] and record["request"]["model"] == "gemma4:26b-mlx":
            key = ChatRequest.model_validate(
                {**record["request"], **record["request"].get("options", {})}
            ).cache_key("cd" * 32)
            legacy.put(key, record["request"], record["response"])
        else:
            legacy.put(path.stem, record["request"], record["response"])
    config.replay.dir = legacy.root
    entry, n = publish(finished.run_dir, tmp_path / "gallery", config, http=arxiv.client())
    assert n == len(list((entry / "cassettes").glob("*/*.json"))) > 10


def test_publish_refuses_unfinished_or_unrecorded_runs(fake, arxiv, config, tmp_path):
    fake.chat_handler = agentic_chat
    paused = run(config, ref="2401.00001", client=fake.client(), http=arxiv.client())
    with pytest.raises(PublishError, match="has not finished"):
        publish(paused.run_dir, tmp_path / "g", config)
    done = run(config, ref="2401.00001", client=fake.client(), http=arxiv.client(), approve=True)
    shutil.rmtree(config.replay.dir)
    with pytest.raises(PublishError, match="does not replay.*CassetteMissError"):
        publish(done.run_dir, tmp_path / "g", config, http=arxiv.client())
    (done.run_dir / "trace.jsonl").write_text("")
    with pytest.raises(PublishError, match="no completed run"):
        publish(done.run_dir, tmp_path / "g", config)


def test_publish_local_pdf_hides_the_local_path(fake, config, tmp_path):
    fake.chat_handler = agentic_chat
    pdf = make_pdf(tmp_path / "My Paper.pdf")
    kw = {"pdf": pdf, "title": "Mine", "client": fake.client()}
    run(config, **kw)
    done = run(config, approve=True, **kw)
    entry, _ = publish(done.run_dir, tmp_path / "gallery", config, include_pdf=True)
    assert (entry / "my-paper.pdf").exists()
    assert Meta.model_validate_json((entry / "meta.json").read_text()).source == "pdf"
    assert str(tmp_path) not in (entry / "trace.jsonl").read_text()


def test_verify_reproduces_every_artifact_from_cassettes(entry, config, arxiv):
    report = verify(entry, Config(), http=arxiv.client())
    assert report.ok, report
    assert "05_factcheck.json" in report.identical and len(report.identical) == 9


def test_verify_reports_differences_and_missing_cassettes(entry, arxiv):
    (entry / "02_claims.json").write_text("{}")
    report = verify(entry, Config(), http=arxiv.client())
    assert report.different == ["02_claims.json"] and not report.ok
    shutil.rmtree(entry / "cassettes")
    report = verify(entry, Config(), http=arxiv.client())
    assert "CassetteMissError" in report.error and not report.ok


def test_verify_local_pdf_entry(fake, config, tmp_path):
    fake.chat_handler = agentic_chat
    kw = {"pdf": make_pdf(tmp_path / "mine.pdf"), "title": "Mine", "client": fake.client()}
    run(config, **kw)
    done = run(config, approve=True, **kw)
    entry, _ = publish(done.run_dir, tmp_path / "gallery", config, include_pdf=True)
    assert verify(entry, Config()).ok


def test_paper_context_describes_the_audit(entry):
    check = json.loads((entry / "05_factcheck.json").read_text())
    first = check["report"]["rounds"][0]
    first[0] |= {"verdict": "unsupported", "reason": "made up"}
    first[1] |= {"problems": ["number(s) 7 not in the cited evidence"]}
    rewritten = dict(first[1], text="A better bullet", problems=[])
    failing_again = dict(first[2], verdict="partial")
    check["report"]["rounds"].append(
        [first[0] | {"verdict": "supported"}, rewritten, failing_again, first[3]]
    )
    first[2] |= {"verdict": "partial"}
    check["report"]["dropped_slides"] = [4]
    first[3] |= {"verdict": "unsupported"}
    (entry / "05_factcheck.json").write_text(json.dumps(check))
    ctx = paper_context(entry)
    outcomes = [a["outcome"] for a in ctx["audit"]]
    assert outcomes == ["passed", "rewritten: A better bullet", "dropped", "slide dropped"]
    assert ctx["route"].paper_type == "method" and ctx["post"] is not None
    assert ctx["slides"][0].bullets[0]["evidence"] and len(ctx["flows"].details) == 2


def test_build_site(entry, tmp_path):
    judges = tmp_path / "judges.json"
    judges.write_text(json.dumps([{
        "model": "gemma4:26b-mlx", "n": 10, "accuracy": 0.8, "kappa": 0.6,
        "accuracy_binary": 0.9, "kappa_binary": 0.7, "confusion": {},
    }]))  # fmt: skip
    out = tmp_path / "site"
    out.mkdir()
    (out / "stale.html").write_text("x")
    metrics = build_site(entry.parent, out, judges)
    assert [m.paper_id for m in metrics] == ["2401.00001"]
    assert not (out / "stale.html").exists() and (out / ".nojekyll").exists()
    index = (out / "index.html").read_text()
    assert 'href="2401.00001/"' in index and "gemma4:26b-mlx" in index and "0.70" in index
    page = (out / "2401.00001" / "index.html").read_text()
    assert "A Test Paper" in page and "Every bullet and its evidence" in page
    assert "overview/page-01.png" in page and "post/page-01.png" in page
    assert "Where would linear attention help your models?" in page  # the post text to copy
    assert "Detail A:" in page  # what the diagrams were checked against
    assert (out / "2401.00001" / "overview" / "page-05.png").exists()
    assert (out / "2401.00001" / "trace.html").exists()
    assert (out / "2401.00001" / "overview.pdf").exists() and (
        out / "2401.00001" / "post.pdf"
    ).exists()
    assert "2401.00001/overview/page-03.png" in index  # the data flow is the thumbnail


def test_build_site_without_entries(tmp_path):
    assert build_site(tmp_path / "none", tmp_path / "site") == []
    assert "No published papers yet" in (tmp_path / "site" / "index.html").read_text()


def test_verify_local_pdf_without_pdf_or_url_reports_it(fake, config, tmp_path):
    fake.chat_handler = agentic_chat
    kw = {"pdf": make_pdf(tmp_path / "mine.pdf"), "title": "Mine", "client": fake.client()}
    run(config, auto_approve=True, **kw)
    with pytest.raises(PublishError, match="no PDF published and no URL"):
        publish(config.tracing.runs_dir / "mine", tmp_path / "gallery", config)


def test_publish_keeps_only_the_cassettes_replay_needs(finished, config, arxiv, tmp_path):
    # a discarded call (e.g. before an outline edit) is in the trace but not needed
    extra = config.replay.dir / "ab" / ("ab" + "1" * 62 + ".json")
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text('{"key": "x", "request": {"messages": []}, "response": {}}')
    span = {"trace_id": "t", "span_id": "s", "parent_id": None, "name": "llm.chat",
            "key": extra.stem, "status": "ok", "latency_ms": 1.0,
            "start_ts": "2026-09-26T10:00:00+00:00"}  # fmt: skip
    finished.trace.write_text(finished.trace.read_text() + json.dumps(span) + "\n")
    entry, n = publish(finished.run_dir, tmp_path / "gallery", config, http=arxiv.client())
    assert not (entry / "cassettes" / "ab").exists()
    assert n == len(list((entry / "cassettes").glob("*/*.json"))) > 10
