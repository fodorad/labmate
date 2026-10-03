"""A fake model that plays every cv2job role, driven by the example CV and posting."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from labmate.cv2job.steps import load_cv
from tests.conftest import reply

EXAMPLES = Path(__file__).parent.parent.parent / "examples" / "cv2job"
NO_ANSWER = "(the candidate gave no answer)"
FILLER = {"experience", "years", "least", "strong", "skills"}


@pytest.fixture
def cv():
    return load_cv(EXAMPLES / "cv.yaml")


@pytest.fixture
def job_text() -> str:
    return (EXAMPLES / "job.txt").read_text()


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{5,}", text.lower()))


def tool_call(body: dict[str, Any], name: str, **arguments: Any) -> dict[str, Any]:
    call = {"function": {"name": name, "arguments": arguments}}
    message = {"role": "assistant", "content": "", "tool_calls": [call]}
    return {"model": body["model"], "message": message}


def gap_agent(body: dict[str, Any]) -> dict[str, Any]:
    """Search the CV, ask about Kubernetes, report what the answer shows, then stop."""
    tools = [m["content"] for m in body["messages"] if m["role"] == "tool"]
    system = next(m["content"] for m in body["messages"] if m["role"] == "system")
    kube = re.search(r"^(q\d+): .*Kubernetes", system, re.MULTILINE).group(1)
    if not tools:
        return tool_call(body, "search_cv", query="kubernetes")
    if len(tools) == 1:
        return tool_call(body, "ask_candidate", question="Have you used Kubernetes in production?")
    if len(tools) == 2:
        answered = tools[1] != NO_ANSWER
        status = "covered" if answered else "gap"
        return tool_call(body, "report_finding", requirement_id=kube, status=status,
                         from_answer=answered)  # fmt: skip
    return reply(body, "Checked the open requirements.")


def read_posting(prompt: str) -> str:
    posting = prompt.split("Job posting:")[1].split('- "position"')[0]
    must, nice = posting.split("plus")
    lines = [(x[2:].strip(), True) for x in must.splitlines() if x.startswith("- ")]
    lines += [(x[2:].strip(), False) for x in nice.splitlines() if x.startswith("- ")]
    requirements = [{"text": t, "quote": t, "must": m} for t, m in lines]
    return json.dumps({"position": "Machine Learning Engineer", "company": "Contoso Vision",
                       "requirements": requirements})  # fmt: skip


def match_bullets(prompt: str) -> str:
    wanted = words(prompt.split("From the posting")[0]) - FILLER
    bullets = re.findall(r"^\[(r\d\.\d)\] [^:]*: (.*)$", prompt, re.MULTILINE)
    ids = [i for i, text in bullets if wanted & words(text)]
    return json.dumps({"bullet_ids": ids[:2], "skills": []})


def keep_first_bullets(prompt: str) -> str:
    bullets = re.findall(r"^\[(r\d\.\d)\] (.*)$", prompt, re.MULTILINE)
    kept = [{"source_id": i, "text": t} for i, t in bullets[:2]]
    return json.dumps({"role_id": "x", "bullets": kept})


def highlight_first_evidence(prompt: str) -> str:
    items = []
    for rid, evidence in re.findall(r"^(q\d+) \(.*\):\n((?:  - .*\n?)+)", prompt, re.MULTILINE):
        first = re.search(r"  - (?!skill:)(.*)", evidence)
        reason = first.group(1) if first else "I have used it."
        items.append({"requirement_id": rid, "skill": "Experience", "reason": reason})
    return json.dumps({"items": items})


def cv_chat(body: dict[str, Any]) -> dict[str, Any]:
    if body.get("tools"):
        return gap_agent(body)
    prompt = body["messages"][-1]["content"]
    for marker, answer in [
        ("You read a job posting", read_posting),
        ("You check whether a CV shows", match_bullets),
        ("You tailor one role", keep_first_bullets),
        ("You write the highlights", highlight_first_evidence),
    ]:
        if marker in prompt:
            return reply(body, answer(prompt))
    raise AssertionError(f"unexpected prompt: {prompt[:80]}")


@pytest.fixture
def model(fake):
    fake.chat_handler = cv_chat
    return fake
