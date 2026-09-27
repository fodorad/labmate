"""Live dashboard for the ask agent (NiceGUI): chat on the left, the agent at work on the right.

- **Graph:** the Mermaid overview, nodes turning green as they run (research and verify
  subgraphs included), the latest one outlined.
- **Models:** which model answered last, calls per model, tokens, time, and how many
  calls came from cassettes; with a live Ollama, the models loaded right now.
- **Events:** the agent's reasoning trail (understood as, planned queries, retrieved,
  graded, rewritten, conflicts, verified) and the latest retrieved chunks with scores.
- **Demo mode** (``--demo``): replay mode with a pause per step, so a recorded session
  plays back at a watchable pace on any laptop, without Ollama.

The logic lives in :mod:`labmate.ask.live` (tested); this module is only UI wiring.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from nicegui import app, ui

from labmate.ask.agent import build_agent, parse_answer
from labmate.ask.graph import compile_graph
from labmate.ask.live import LiveView, model_stats
from labmate.ask.schemas import Answer
from labmate.ask.session import AskSession, open_ask
from labmate.config import Config, ReplayMode
from labmate.core.llm.client import OllamaClient
from labmate.core.theme import Theme
from labmate.core.tracing import read_trace

SAMPLE_QUESTIONS = [
    "What are the thesis points of the dissertation?",
    "How does BlinkLinMulT differ from LinMulT?",
    "Which datasets are used to evaluate blink detection, and with what results?",
]
"""Shown as chips when the library has no golden set."""


def _golden_questions(config: Config) -> list[str]:
    path = config.ask.library / "golden.yaml"
    if not path.exists():
        return SAMPLE_QUESTIONS
    from labmate.ask.evals import load_golden  # noqa: PLC0415

    return [g.question for g in load_golden(path) if not g.abstain][:6] or SAMPLE_QUESTIONS


def serve(
    config: Config,
    port: int = 8080,
    demo: bool = False,
    pace: float = 0.6,
    client: OllamaClient | None = None,
) -> None:  # pragma: no cover - UI wiring, exercised by hand
    """Start the dashboard.

    Args:
        config: Loaded configuration.
        port: HTTP port.
        demo: Replay recorded sessions (no Ollama), pausing ``pace`` s per step.
        pace: Seconds per step in demo mode.
        client: Ollama client (live mode).
    """
    mode = ReplayMode.REPLAY if demo else None
    s: AskSession = open_ask(config, mode, client, label="ask-dashboard")
    graph = compile_graph(s)
    agent = build_agent(s)
    trace = s.tracer.path or config.ask.library / "trace.jsonl"
    trace_start = len(read_trace(trace)) if trace.exists() else 0
    theme = Theme()
    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    ollama = client if (client is not None and not demo) else None

    def stream_graph(payload: Any, thread: str) -> None:
        cfg = {"configurable": {"thread_id": thread}}
        try:
            for item in graph.stream(  # type: ignore[call-overload]
                payload, cfg, stream_mode=["updates", "custom"], subgraphs=True
            ):
                events.put(("graph", item))
                if demo:
                    time.sleep(pace)
        except Exception as e:  # noqa: BLE001 - shown in the UI
            events.put(("error", str(e)))
        events.put(("done", None))

    def stream_agent(question: str) -> None:
        try:
            final = ""
            for update in agent.stream(
                {"messages": [HumanMessage(question)]},
                {"recursion_limit": 16},
                stream_mode="updates",
            ):
                for node, value in update.items():
                    for m in value.get("messages", []):
                        calls = getattr(m, "tool_calls", None) or []
                        for c in calls:
                            events.put(("log", f"{node}: {c['name']}({c['args']})"))
                        if node == "model" and not calls:
                            final = str(m.content)
                        if node == "tools":
                            events.put(("log", f"tool result: {str(m.content)[:160]}…"))
                if demo:
                    time.sleep(pace)
            events.put(("answer", parse_answer(s, question, final)))
        except Exception as e:  # noqa: BLE001 - shown in the UI
            events.put(("error", str(e)))
        events.put(("done", None))

    @ui.page("/")
    def index() -> None:
        view = LiveView()
        ui.add_head_html(
            "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700"
            "&display=swap' rel='stylesheet'>"
            f"<style>body{{font-family:Inter,sans-serif;background:{theme.background};"
            f"color:{theme.text}}} .muted{{color:{theme.muted}}}</style>"
        )
        with ui.header().style(f"background:{theme.surface};color:{theme.text}"):
            ui.label("labmate · ask my research").classes("text-lg font-bold")
            ui.label("demo: replaying recorded sessions" if demo else s.mode.value).classes("muted")
        with ui.row().classes("w-full no-wrap items-start gap-4"):
            with ui.column().classes("w-5/12"):
                with ui.row().classes("w-full items-center"):
                    agent_choice = ui.toggle(["graph", "prebuilt"], value="graph")
                    thread = ui.input("thread", value="default").classes("w-32")
                question = ui.input("Ask about the dissertation and papers").classes("w-full")
                with ui.row():
                    for q in _golden_questions(config):
                        ui.chip(q, on_click=lambda _, q=q: question.set_value(q)).props(
                            "outline clickable"
                        )
                choices = ui.row()
                answer_box = ui.column().classes("w-full")
                ask_button = ui.button("Ask")
            with ui.column().classes("w-7/12"):
                diagram = ui.mermaid(view.mermaid()).classes("w-full")
                with ui.card().classes("w-full"):
                    ui.label("Models").classes("font-bold")
                    models = ui.label("no calls yet").classes("muted")
                with ui.card().classes("w-full"):
                    ui.label("Agent trail").classes("font-bold")
                    trail = ui.log(max_lines=200).classes("w-full h-48")
                hits_table = ui.table(
                    columns=[
                        {"name": "id", "label": "chunk", "field": "id"},
                        {"name": "tier", "label": "tier", "field": "tier"},
                        {"name": "score", "label": "score", "field": "score"},
                        {"name": "text", "label": "text", "field": "text", "align": "left"},
                    ],
                    rows=[],
                ).classes("w-full")

        def show_answer(answer: Answer) -> None:
            answer_box.clear()
            with answer_box:
                ui.markdown(answer.text)
                for c in answer.citations:
                    with ui.expansion(f"[{c.n}] {c.label}").classes("w-full"):
                        ui.label(c.text).classes("muted")
                for x in answer.conflicts:
                    ui.label(f"Note: {x.topic}: dissertation {x.dissertation_value}, "
                             f"other source {x.other_value}").classes("muted")  # fmt: skip
                if answer.dropped:
                    ui.label(f"{answer.dropped} unsupported sentence(s) removed").classes("muted")

        def start(payload: Any) -> None:
            ask_button.disable()
            choices.clear()
            if agent_choice.value == "prebuilt":
                threading.Thread(target=stream_agent, args=(question.value,), daemon=True).start()
            else:
                threading.Thread(
                    target=stream_graph, args=(payload, thread.value or "default"), daemon=True
                ).start()

        def ask_clicked() -> None:
            view.__init__()  # type: ignore[misc]
            trail.clear()
            answer_box.clear()
            start({"question": question.value})

        ask_button.on_click(ask_clicked)

        def resume(option: str) -> None:
            view.interrupt = None
            start(Command(resume=option))

        def drain() -> None:
            changed = False
            while not events.empty():
                kind, item = events.get_nowait()
                if kind == "graph":
                    before = len(view.log)
                    view.feed(item)
                    for line in view.log[before:]:
                        trail.push(line)
                    changed = True
                elif kind == "log":
                    trail.push(item)
                elif kind == "answer":
                    show_answer(item)
                elif kind == "error":
                    trail.push(f"error: {item}")
                elif kind == "done":
                    ask_button.enable()
                    if view.answer is not None:
                        show_answer(view.answer)
                    if view.interrupt:
                        with choices:
                            ui.label("Which one do you mean?")
                            for option in view.interrupt["options"]:
                                ui.button(option, on_click=lambda _, o=option: resume(o))
            if changed:
                diagram.set_content(view.mermaid())
                hits_table.rows = view.hits
                hits_table.update()
            spans = read_trace(trace)[trace_start:] if trace.exists() else []
            stats = model_stats(spans)
            loaded = ""
            if ollama is not None:
                try:
                    loaded = " · loaded now: " + ", ".join(ollama.running_models())
                except Exception:  # noqa: BLE001 - informative only
                    loaded = ""
            per_model = ", ".join(f"{m}: {n}" for m, n in stats.calls.items()) or "none"
            models.set_text(
                f"last: {stats.last_model or '-'} · calls {per_model} · embeddings {stats.embeds}"
                f" · tokens {stats.tokens_in} in / {stats.tokens_out} out · "
                f"{stats.seconds:.0f} s · from cassettes {stats.cached}{loaded}"
            )

        ui.timer(0.3, drain)

    app.on_shutdown(s.close)
    ui.run(port=port, title="labmate · ask", reload=False, show=False)
