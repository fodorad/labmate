"""Single-model measurements: loading, speed, structured output, tool calls and judging.

Each model is unloaded first, so the first call pays the real cold-load cost; the speeds come from
Ollama's own timings (see :mod:`labmate.core.usage`). The judge test sends the planted mistakes of
:mod:`labmate.bench.judge_cases` through the pipeline's real judge prompt.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel

from labmate.bench.judge_cases import CASES, JudgeCase
from labmate.config import CacheConfig, Config
from labmate.core.chat import chat_model
from labmate.core.factcheck import judge_card
from labmate.core.lc import prompt, structured
from labmate.core.schemas import Bullet, Card, ClaimCard
from labmate.core.structured import StructuredOutputError
from labmate.core.usage import UsageCollector, ollama_resident

MAX_BULLETS = 4
"""Bullets per card, as in a real run."""

PASSAGE = (
    "Linear attention replaces the softmax kernel with a feature map, so the cost of attention "
    "grows linearly with the sequence length instead of quadratically. "
) * 40
"""A prompt of about 1200 tokens, to measure how fast a model reads."""

CITIES = [
    ("Budapest has about 1.7 million inhabitants.", "Budapest", 1.7),
    ("Vienna is home to roughly 2.0 million people.", "Vienna", 2.0),
    ("Prague has around 1.3 million residents.", "Prague", 1.3),
    ("Warsaw counts nearly 1.8 million inhabitants.", "Warsaw", 1.8),
    ("Zagreb has about 0.8 million people.", "Zagreb", 0.8),
]
"""Sentences with the city and population (millions) a model should extract."""


class City(BaseModel):
    """The schema of the structured-output test."""

    name: str
    population_millions: float


@dataclass
class ModelResult:
    """What one model did in the micro benchmark.

    Attributes:
        model: Model tag.
        size_gb: Size on disk (about what it takes in memory).
        load_s: Cold-load seconds (first call after unloading).
        prompt_tps: Prompt tokens read per second.
        decode_tps: Tokens written per second.
        structured_ok: Valid structured answers on the first try, out of ``len(CITIES)``.
        tools_ok: Correct tool calls out of 3 (``None`` if not tested).
        judge: The judge test result (``None`` if not run).
        error: What went wrong, if anything.
    """

    model: str
    size_gb: float = 0.0
    load_s: float = 0.0
    prompt_tps: float = 0.0
    decode_tps: float = 0.0
    structured_ok: int = 0
    tools_ok: int | None = None
    judge: JudgeResult | None = None
    error: str = ""


@dataclass
class JudgeResult:
    """How a model judged the planted mistakes.

    Attributes:
        caught: Mistakes it did not call ``supported``.
        mistakes: Mistakes in the test.
        false_alarms: Clean statements it did not call ``supported``.
        clean: Clean statements in the test.
        seconds_per_verdict: Seconds per statement judged.
        missed: The mistakes it let through, by kind.
    """

    caught: int
    mistakes: int
    false_alarms: int
    clean: int
    seconds_per_verdict: float
    missed: list[str] = field(default_factory=list)


def unload_all(host: str, transport: httpx.BaseTransport | None = None) -> None:
    """Unload every model Ollama has loaded, so the next call is a cold start.

    Args:
        host: The Ollama server address.
        transport: An httpx transport to reach it through (tests).
    """
    with httpx.Client(transport=transport, timeout=60) as http:
        for loaded in http.get(f"{host}/api/ps").json().get("models", []):
            http.post(f"{host}/api/generate", json={"model": loaded["name"], "keep_alive": 0})


def model_sizes(host: str, transport: httpx.BaseTransport | None = None) -> dict[str, float]:
    """The size in gigabytes of every installed model.

    Args:
        host: The Ollama server address.
        transport: An httpx transport to reach it through (tests).

    Returns:
        Model tag to gigabytes.
    """
    with httpx.Client(transport=transport, timeout=30) as http:
        models = http.get(f"{host}/api/tags").json().get("models", [])
    return {m["name"]: m["size"] / 1e9 for m in models}


JUDGE_MAX_TOKENS = 800
"""Most tokens a judge may write for one card. A reasoning model that loops would otherwise stream
for as long as it likes (the client timeout only fires when the stream goes quiet); cut off, its
reply is not valid JSON and the judge test fails with an error instead of hanging."""


def _bench_config(config: Config, collector: UsageCollector) -> Config:
    """A copy of the configuration that reports to ``collector`` and does not use the cache."""
    return config.model_copy(
        update={"cache": CacheConfig(enabled=False), "callbacks": [collector]}, deep=True
    )


def judge_test(judge: Any, config: RunnableConfig | None = None) -> JudgeResult:
    """Send the planted mistakes through the pipeline's real judge prompt.

    The statements are grouped by their evidence into cards, as a real run would, one judge call
    per card.

    Args:
        judge: The judge model.
        config: The calling step's config (callbacks).

    Returns:
        How many mistakes were caught and how many clean statements were wrongly doubted.
    """
    by_evidence: dict[str, list[JudgeCase]] = {}
    for case in CASES:
        by_evidence.setdefault(case.evidence, []).append(case)
    groups = [
        (evidence, cases[i : i + MAX_BULLETS])
        for evidence, cases in by_evidence.items()
        for i in range(0, len(cases), MAX_BULLETS)
    ]
    caught = alarms = 0
    missed: list[str] = []
    started = time.perf_counter()
    for number, (evidence, cases) in enumerate(groups, start=1):
        claim = ClaimCard(id="c01", claim=evidence, evidence_quote=evidence, kind="method",
                          section="", page=1)  # fmt: skip
        bullets = [Bullet(text=c.statement, claim_ids=["c01"]) for c in cases]
        judged = judge_card(
            Card(title=f"case {number}", bullets=bullets), {"c01": claim}, judge, config
        )
        verdicts = {v.bullet: v.verdict for v in judged.verdicts}
        for i, case in enumerate(cases, start=1):
            doubted = verdicts[i] != "supported"
            if case.right and doubted:
                alarms += 1
            if not case.right and doubted:
                caught += 1
            if not case.right and not doubted:
                missed.append(case.kind)
    return JudgeResult(
        caught=caught,
        mistakes=sum(not c.right for c in CASES),
        false_alarms=alarms,
        clean=sum(c.right for c in CASES),
        seconds_per_verdict=(time.perf_counter() - started) / len(CASES),
        missed=missed,
    )


def measure_model(config: Config, tag: str, tools: bool, judging: bool) -> ModelResult:
    """Cold-load a model and measure it.

    Args:
        config: Loaded configuration.
        tag: Model tag.
        tools: Also test tool calling (for models that will be a writer or an agent).
        judging: Also run the judge test.

    Returns:
        The measurements; ``error`` is set if the model failed in a way that stops the test.
    """
    unload_all(config.ollama.host)
    collector = UsageCollector(ollama_resident(config.ollama.host))
    cfg = _bench_config(config, collector)
    model = chat_model(cfg, tag)
    result = ModelResult(tag, size_gb=model_sizes(config.ollama.host).get(tag, 0.0))
    try:
        model.invoke("Say OK.")  # the cold start: its load time is what a first request pays
        result.load_s = collector.summary().load_s
        probe = UsageCollector()
        reader = chat_model(_bench_config(config, probe), tag).model_copy(
            update={"num_predict": 96}
        )
        reader.invoke(f"{PASSAGE}\nSummarise the passage in two sentences.")
        speed = probe.summary()
        result.prompt_tps, result.decode_tps = speed.prompt_tps, speed.decode_tps
        for sentence, name, millions in CITIES:
            try:
                chain = prompt("Extract the city and its population in millions from: {text}")
                out = (chain | structured(model, City, max_retries=0)).invoke({"text": sentence})
                result.structured_ok += (
                    out.name == name and abs(out.population_millions - millions) < 0.05
                )
            except StructuredOutputError:
                continue
        if tools:
            result.tools_ok = _tool_calls(model)
        if judging:
            result.judge = judge_test(model.model_copy(update={"num_predict": JUDGE_MAX_TOKENS}))
    except (
        httpx.HTTPError,
        ValueError,
        RuntimeError,
    ) as e:  # a model that cannot be reached or served
        result.error = f"{type(e).__name__}: {e}"[:200]
    return result


def _tool_calls(model: Any) -> int:
    """How many of three requests to use a tool produce the right call."""

    @tool
    def lookup(term: str) -> str:
        """Look a term up in the library.

        Args:
            term: What to look up.
        """
        return term

    correct = 0
    for term in ("blink", "transformer", "speech"):
        reply = model.bind_tools([lookup]).invoke(f"Use the lookup tool to look up '{term}'.")
        calls = reply.tool_calls
        correct += bool(calls) and calls[0]["name"] == "lookup"
    return correct


def to_dict(result: ModelResult) -> dict[str, Any]:
    """A result as plain data, for the JSON file.

    Args:
        result: One model's measurements.

    Returns:
        Nested dictionaries and lists.
    """
    return asdict(result)


def from_dict(data: dict[str, Any]) -> ModelResult:
    """A result read back from the JSON file.

    Args:
        data: One entry of ``micro.json``.

    Returns:
        The measurements.
    """
    data = dict(data)
    judge = data.pop("judge")
    return ModelResult(**data, judge=JudgeResult(**judge) if judge else None)
