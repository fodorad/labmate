"""Capability probe: verify what the local models can actually do before relying on it.

The ``-mlx`` builds run on Ollama's MLX backend, whose feature coverage (structured output,
tool calling, vision, seeded determinism) must be checked, not assumed. The probe also
measures cold-load time and throughput, which drive the phase design (only one large model
fits in 32 GB), and benchmarks the candidate image models (plan Q5).

Probe calls always go straight to Ollama: no cassettes, because measuring the model is the
point.
"""

from __future__ import annotations

import base64
import logging
import struct
import time
import zlib
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from paper2carousel.llm.structured import parse_structured, schema_instruction
from paper2carousel.llm.types import (
    ChatRequest,
    ChatResponse,
    ImageRequest,
    ImageResponse,
    Message,
)

log = logging.getLogger(__name__)
"""Progress messages; the CLI prints them so long runs don't look hung."""

PROBE_MAX_TOKENS = 512
"""Generation cap for every probe call, so a rambling (or thinking) model can't stall it."""

UNLOAD_WAIT_S = 30.0
"""How long to wait for a model to leave memory after ``keep_alive=0``."""

UNLOAD_POLL_S = 0.5
"""Polling interval while waiting for an unload."""

PROBE_PASSAGE = (
    "We introduce LinMulT, a multimodal transformer with linear-complexity attention. "
    "On the CMU-MOSEI benchmark it improves binary sentiment accuracy from 82.1% to 84.6% "
    "while reducing memory use by 38% compared to the quadratic-attention baseline. "
    "A limitation is that it was only evaluated on English-language videos."
)
"""Short synthetic passage used by the text checks (numbers are made up for the probe)."""

VISION_COLOR = (220, 30, 30)
"""RGB colour of the solid test image used by the vision check (red)."""

VISION_SIZE = 256
"""Side length of the vision test image; large enough for any VLM's minimum patch grid."""

FIGURE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "use_paper_figure",
        "description": "Place one of the paper's figures on the current slide.",
        "parameters": {
            "type": "object",
            "properties": {
                "figure_id": {"type": "string", "description": "Figure id, e.g. 'fig2'."}
            },
            "required": ["figure_id"],
        },
    },
}
"""Tool definition mirroring the visuals agent's real ``use_paper_figure`` tool."""


class ProbeClaim(BaseModel):
    """Schema used to test constrained decoding (mirrors the real claim cards)."""

    claim: str
    evidence_quote: str
    kind: Literal["contribution", "result", "method", "limitation"]


class ProbeClient(Protocol):
    """The subset of :class:`~paper2carousel.llm.client.OllamaClient` the probe needs."""

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Run a chat request."""
        ...

    def generate_image(self, request: ImageRequest) -> ImageResponse:
        """Run an image request."""
        ...

    def unload(self, model: str, wait_s: float = 0.0, poll_s: float = 0.5) -> bool:
        """Unload a model, optionally waiting until it is gone."""
        ...

    def running_models(self) -> list[str]:
        """List loaded models."""
        ...

    def show(self, model: str) -> dict[str, Any]:
        """Return model metadata."""
        ...


class CheckResult(BaseModel):
    """Outcome of one probe check.

    Attributes:
        model: Model under test.
        check: Check name.
        passed: True/False, or None for informational checks.
        detail: Human-readable explanation.
        metrics: Numeric or short string measurements.
    """

    model: str
    check: str
    passed: bool | None
    detail: str = ""
    metrics: dict[str, float | int | str] = Field(default_factory=dict)


class ProbeReport(BaseModel):
    """All probe results plus environment info."""

    ollama_version: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    results: list[CheckResult] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the report as a markdown table.

        Returns:
            Markdown text.
        """
        icon = {True: "✅", False: "❌", None: "ℹ️"}
        lines = [
            "# Capability probe",
            "",
            f"Ollama `{self.ollama_version}` · {self.created_at}",
            "",
            "| Model | Check | Result | Detail | Metrics |",
            "|---|---|---|---|---|",
        ]
        for r in self.results:
            metrics = ", ".join(f"{k}={v}" for k, v in r.metrics.items())
            detail = r.detail.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{r.model}` | {r.check} | {icon[r.passed]} | {detail} | {metrics} |")
        return "\n".join(lines) + "\n"


def solid_png(rgb: tuple[int, int, int], size: int = 64) -> bytes:
    """Encode a solid-colour RGB PNG without any imaging library.

    Args:
        rgb: Colour.
        size: Width and height in pixels.

    Returns:
        PNG file bytes.
    """

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\x00" + bytes(rgb) * size  # filter byte 0 + pixels
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(row * size))
        + chunk(b"IEND", b"")
    )


def _think_flag(capabilities: Iterable[str]) -> bool | None:
    # Disable thinking where supported (faster, cleaner JSON); leave unset otherwise,
    # because some servers reject the flag for models without thinking support.
    return False if "thinking" in capabilities else None


def _safe_unload(client: ProbeClient, model: str) -> None:
    try:
        client.unload(model, wait_s=UNLOAD_WAIT_S, poll_s=UNLOAD_POLL_S)
    except Exception:  # best effort; the next load will evict it anyway
        pass


def _informational(result: CheckResult) -> CheckResult:
    """Downgrade a result to informational: it is reported but cannot fail the probe."""
    if result.passed is None:
        return result
    verdict = "would pass" if result.passed else "would fail"
    return result.model_copy(update={"passed": None, "detail": f"({verdict}) {result.detail}"})


def _guarded(
    model: str, check: str, fn: Callable[[], CheckResult], required: bool = True
) -> CheckResult:
    log.info("  %-28s ...", check)
    t0 = time.perf_counter()
    try:
        result = fn()
    except Exception as e:  # a failing check must not abort the whole probe
        result = CheckResult(
            model=model, check=check, passed=False, detail=f"{type(e).__name__}: {e}"
        )
    if not required:
        result = _informational(result)
    status = {True: "ok", False: "FAIL", None: "info"}[result.passed]
    log.info("  %-28s %-4s %6.1fs  %s", check, status, time.perf_counter() - t0, result.detail)
    return result


def declared_capabilities(client: ProbeClient, model: str) -> list[str]:
    """Return the capabilities the server declares for a model (e.g. ``vision``, ``tools``).

    Args:
        client: Probe client.
        model: Model tag.

    Returns:
        Capability names; empty if the server declares none or the lookup fails.
    """
    try:
        return list(client.show(model).get("capabilities", []))
    except Exception:
        return []


def check_structured_output(
    client: ProbeClient, model: str, think: bool | None, prompted: bool = False, n: int = 10
) -> CheckResult:
    """Check that the model returns schema-valid JSON across seeds.

    Uses temperature 0.7 and ``n`` different seeds to stress the constraint, not the prompt.
    Always sends ``format=`` (constrained decoding). With ``prompted=True`` it also puts the
    schema in a system message, which is what the pipeline does for backends that ignore
    ``format=``. Counts strict parses and lenient ones (fences/prose stripped).

    Args:
        client: Probe client.
        model: Model tag.
        think: Thinking flag to send.
        prompted: Also instruct the model with the schema in a system message.
        n: Number of attempts.

    Returns:
        Pass if all ``n`` responses validate strictly against :class:`ProbeClaim`.
    """
    strict, lenient, tps = 0, 0, []
    sample = ""
    system = [Message(role="system", content=schema_instruction(ProbeClaim))] if prompted else []
    for seed in range(n):
        response = client.chat(
            ChatRequest(
                model=model,
                messages=[
                    *system,
                    Message(
                        role="user",
                        content="Extract the main result claim from this passage, with a "
                        f"verbatim evidence quote.\n\n{PROBE_PASSAGE}",
                    ),
                ],
                format=ProbeClaim.model_json_schema(),
                temperature=0.7,
                seed=seed,
                think=think,
                num_predict=PROBE_MAX_TOKENS,
            )
        )
        tps.append(response.usage.tokens_per_s)
        _, mode = parse_structured(response.content, ProbeClaim)
        strict += mode == "strict"
        lenient += mode is not None
        if mode != "strict" and not sample:
            sample = response.content[:120]
    detail = f"strict {strict}/{n}, lenient {lenient}/{n}"
    if sample:
        detail += f"; first non-strict output: {sample!r}"
    return CheckResult(
        model=model,
        check="structured output (schema in prompt)" if prompted else "structured output",
        passed=strict == n,
        detail=detail,
        metrics={"gen_tok_per_s": round(sum(tps) / len(tps), 1)},
    )


def check_tool_calling(client: ProbeClient, model: str, think: bool | None) -> CheckResult:
    """Check that the model emits a well-formed call to the figure tool.

    Args:
        client: Probe client.
        model: Model tag.
        think: Thinking flag to send.

    Returns:
        Pass if the first tool call is ``use_paper_figure(figure_id="fig3")``.
    """
    response = client.chat(
        ChatRequest(
            model=model,
            messages=[
                Message(
                    role="user",
                    content="Put Figure 3 of the paper on this slide. "
                    "Available figure ids: fig1, fig2, fig3.",
                )
            ],
            tools=[FIGURE_TOOL],
            think=think,
            num_predict=PROBE_MAX_TOKENS,
        )
    )
    if not response.tool_calls:
        return CheckResult(
            model=model,
            check="tool calling",
            passed=False,
            detail=f"no tool call; text: {response.content[:80]!r}",
        )
    call = response.tool_calls[0].function
    ok = call.name == "use_paper_figure" and call.arguments.get("figure_id") == "fig3"
    return CheckResult(
        model=model, check="tool calling", passed=ok, detail=f"{call.name}({call.arguments})"
    )


def check_vision(client: ProbeClient, model: str, think: bool | None) -> CheckResult:
    """Check that the model accepts an image and reads a trivial property of it.

    Args:
        client: Probe client.
        model: Model tag.
        think: Thinking flag to send.

    Returns:
        Pass if the model names the colour of a solid red image.
    """
    image = base64.b64encode(solid_png(VISION_COLOR, size=VISION_SIZE)).decode()
    response = client.chat(
        ChatRequest(
            model=model,
            messages=[
                Message(
                    role="user",
                    content="What colour is this image? Answer with one word.",
                    images=[image],
                )
            ],
            think=think,
            num_predict=PROBE_MAX_TOKENS,
        )
    )
    answer = response.content.strip()
    return CheckResult(
        model=model, check="vision", passed="red" in answer.lower(), detail=f"answer: {answer!r}"
    )


def check_determinism(
    client: ProbeClient, model: str, think: bool | None, runs: int = 3
) -> CheckResult:
    """Check that temperature 0 with a fixed seed gives identical outputs.

    Args:
        client: Probe client.
        model: Model tag.
        think: Thinking flag to send.
        runs: Number of repeated calls.

    Returns:
        Pass if all outputs are identical.
    """
    request = ChatRequest(
        model=model,
        messages=[
            Message(
                role="user",
                content=f"Summarise this in two sentences for a LinkedIn slide:\n\n{PROBE_PASSAGE}",
            )
        ],
        temperature=0.0,
        seed=42,
        think=think,
        num_predict=PROBE_MAX_TOKENS,
    )
    outputs = {client.chat(request).content for _ in range(runs)}
    return CheckResult(
        model=model,
        check="determinism",
        passed=len(outputs) == 1,
        detail=f"{len(outputs)} distinct output(s) over {runs} runs",
    )


def check_load_and_unload(client: ProbeClient, model: str, think: bool | None) -> CheckResult:
    """Measure cold-load time, then verify that ``keep_alive=0`` frees the model.

    Args:
        client: Probe client.
        model: Model tag.
        think: Thinking flag to send.

    Returns:
        Pass if the model leaves ``/api/ps`` within :data:`UNLOAD_WAIT_S`.
    """
    client.unload(model, wait_s=UNLOAD_WAIT_S, poll_s=UNLOAD_POLL_S)
    response = client.chat(
        ChatRequest(
            model=model,
            messages=[Message(role="user", content="Say OK.")],
            think=think,
            num_predict=PROBE_MAX_TOKENS,
        )
    )
    t0 = time.perf_counter()
    gone = client.unload(model, wait_s=UNLOAD_WAIT_S, poll_s=UNLOAD_POLL_S)
    unload_s = round(time.perf_counter() - t0, 1)
    return CheckResult(
        model=model,
        check="cold load + unload",
        passed=gone,
        detail="unloaded" if gone else f"still resident {UNLOAD_WAIT_S:.0f}s after keep_alive=0",
        metrics={"cold_load_s": round(response.usage.load_ms / 1000, 1), "unload_s": unload_s},
    )


def probe_chat_model(client: ProbeClient, model: str, vision: bool) -> list[CheckResult]:
    """Run all chat-model checks for one model, then unload it.

    Only checks the pipeline depends on can fail. Diagnostics are reported as
    informational: constrained decoding via ``format=`` alone (the pipeline always adds the
    schema to the prompt), and vision on models that declare it but don't have the vision role.

    Args:
        client: Probe client.
        model: Model tag.
        vision: Whether this model has the vision role (vision check is required).

    Returns:
        One result per check. Tool calling is reported as skipped for models that declare
        capabilities without ``tools``.
    """
    caps = declared_capabilities(client, model)
    caps_result = CheckResult(
        model=model,
        check="declared capabilities",
        passed=None,
        detail=", ".join(caps) if caps else "none declared",
    )
    think = _think_flag(caps)
    log.info("%s  (capabilities: %s; think=%s)", model, caps_result.detail, think)
    # (name, check, required)
    checks: list[tuple[str, Callable[[], CheckResult], bool]] = [
        ("cold load + unload", lambda: check_load_and_unload(client, model, think), True),
        ("structured output", lambda: check_structured_output(client, model, think), False),
        (
            "structured output (schema in prompt)",
            lambda: check_structured_output(client, model, think, prompted=True),
            True,
        ),
        ("determinism", lambda: check_determinism(client, model, think), True),
    ]
    skipped: list[CheckResult] = []
    if caps and "tools" not in caps:
        skipped.append(
            CheckResult(model=model, check="tool calling", passed=None, detail="not declared")
        )
    else:
        checks.insert(2, ("tool calling", lambda: check_tool_calling(client, model, think), True))
    if vision or "vision" in caps:
        checks.append(("vision", lambda: check_vision(client, model, think), vision))
    results = [caps_result, *skipped]
    for name, fn, required in checks:
        results.append(_guarded(model, name, fn, required))
    _safe_unload(client, model)
    return results


def probe_image_model(
    client: ProbeClient,
    model: str,
    out_dir: Path,
    prompt: str = "Minimal flat illustration of a transformer neural network, soft colours",
    seeds: Sequence[int] = (1, 2),
) -> list[CheckResult]:
    """Benchmark an image model at carousel resolution and check seed determinism.

    Images are saved to ``out_dir`` for visual comparison.

    Args:
        client: Probe client.
        model: Image model tag.
        out_dir: Where to save generated PNGs.
        prompt: Prompt used for all images.
        seeds: Seeds to generate; the first one is generated twice for determinism.

    Returns:
        A speed result and a determinism result.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("%s  (image model; first call includes the cold load)", model)
    safe = model.replace("/", "_").replace(":", "_")

    def gen(seed: int) -> tuple[ImageResponse, float]:
        t0 = time.perf_counter()
        response = client.generate_image(ImageRequest(model=model, prompt=prompt, seed=seed))
        return response, time.perf_counter() - t0

    def speed() -> CheckResult:
        timings, first = [], ""
        for seed in seeds:
            response, seconds = gen(seed)
            timings.append(seconds)
            (out_dir / f"{safe}_seed{seed}.png").write_bytes(response.image_bytes())
            first = first or response.sha256()
        return CheckResult(
            model=model,
            check="image generation 1080x1350",
            passed=True,
            detail=f"{len(seeds)} images saved to {out_dir}",
            metrics={"s_per_image": round(sum(timings) / len(timings), 1), "first_sha": first[:12]},
        )

    def determinism(first_sha: str) -> CheckResult:
        again, _ = gen(seeds[0])
        return CheckResult(
            model=model,
            check="image determinism",
            passed=again.sha256()[:12] == first_sha,
            detail="same seed → same image" if again.sha256()[:12] == first_sha else "differs",
        )

    speed_result = _guarded(model, "image generation 1080x1350", speed)
    results = [speed_result]
    if speed_result.passed:
        first_sha = str(speed_result.metrics["first_sha"])
        results.append(_guarded(model, "image determinism", lambda: determinism(first_sha)))
    _safe_unload(client, model)
    return results


def run_probe(
    client: ProbeClient,
    ollama_version: str,
    chat_models: Sequence[tuple[str, bool]],
    image_models: Sequence[str],
    out_dir: Path,
) -> ProbeReport:
    """Probe models one at a time (one resident model at a time, as in the pipeline).

    Args:
        client: Probe client.
        ollama_version: Server version, recorded in the report.
        chat_models: ``(model, run_vision_check)`` pairs.
        image_models: Image model tags to benchmark.
        out_dir: Output directory for the report and images.

    Returns:
        The full report (also written as ``probe_report.json`` and ``probe_report.md``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for loaded in client.running_models():  # start from a clean memory state
        log.info("unloading %s", loaded)
        client.unload(loaded)
    report = ProbeReport(ollama_version=ollama_version)
    for model, vision in chat_models:
        report.results.extend(probe_chat_model(client, model, vision))
    for model in image_models:
        report.results.extend(probe_image_model(client, model, out_dir / "images"))
    (out_dir / "probe_report.json").write_text(report.model_dump_json(indent=2) + "\n")
    (out_dir / "probe_report.md").write_text(report.to_markdown())
    return report
