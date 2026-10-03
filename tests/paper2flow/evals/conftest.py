"""Fixtures for the evaluation tests: a finished run and a synthetic fact-check audit."""

from __future__ import annotations

from pathlib import Path

import pytest

from labmate.config import Config
from labmate.paper2flow.chain import ARTIFACTS, paper2flow
from labmate.paper2flow.schemas import (
    Bullet,
    BulletCheck,
    Card,
    Cards,
    ClaimCard,
    Claims,
    FactChecked,
    FactCheckReport,
)
from tests.conftest import agentic_chat


@pytest.fixture
def config(tmp_path):
    cfg = Config()
    cfg.cache.path = tmp_path / "cache" / "replies.sqlite"
    cfg.tracing.runs_dir = tmp_path / "runs"
    return cfg


@pytest.fixture
def finished(fake, arxiv, config) -> Path:
    """A finished paper2flow run."""
    fake.chat_handler = agentic_chat
    return paper2flow(config, "2401.00001", web=arxiv.transport(), ollama=fake.transport()).parent


def card(i: int, quote: str) -> ClaimCard:
    return ClaimCard(
        id=f"c{i:02d}",
        claim=quote,
        evidence_quote=quote,
        kind="result",
        section="Results",
        page=3,
    )


def check(card: int, bullet: int, text: str, ids: list[str], verdict: str) -> BulletCheck:
    return BulletCheck(
        card=card, bullet=bullet, text=text, claim_ids=ids, verdict=verdict, reason="r"
    )


def write_audit(run_dir: Path, n_supported: int = 6, n_rejected: int = 4) -> Path:
    """A run directory with claims and a two-round fact-check audit, no model needed."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cards = [card(1, "Accuracy is 84.6%."), card(2, "Memory drops by 38%.")]
    (run_dir / ARTIFACTS["claims"]).write_text(Claims(cards=cards, rejected=[]).model_dump_json())
    round0 = [check(1, i + 1, f"Good bullet {i}", ["c01"], "supported") for i in range(n_supported)]
    round0 += [
        check(2, i + 1, f"WRONG bullet {i}", ["c01", "c02"], "unsupported")
        for i in range(n_rejected)
    ]
    round1 = [*round0[:n_supported], check(2, 1, "Fixed bullet", ["c02"], "supported")]
    report = FactCheckReport(rounds=[round0, round1], dropped=round0[n_supported:])
    final = Cards(cards=[Card(title="t", bullets=[Bullet(text="x", claim_ids=["c01"])])])
    (run_dir / ARTIFACTS["checked"]).write_text(
        FactChecked(cards=final, report=report).model_dump_json()
    )
    return run_dir
