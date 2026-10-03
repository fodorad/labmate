import json
from datetime import date

import pymupdf

from labmate.cli import main
from labmate.cv2job.gaps import find_gaps
from labmate.cv2job.letter import years_sentence
from labmate.cv2job.schemas import Highlight
from labmate.cv2job.steps import match_requirements, read_job, tailor_roles, write_highlights
from tests.conftest import chat, reply
from tests.cv2job.conftest import EXAMPLES

KUBERNETES = {"text": "Kubernetes", "quote": "Experience with Kubernetes in production.",
              "must": True}  # fmt: skip


def pdf_text(path) -> str:
    with pymupdf.open(path) as doc:
        return " ".join(" ".join(page.get_text().split()) for page in doc)


def prepare(cv, model, job_text):
    job = read_job(job_text, chat(model))
    return job, match_requirements(cv, job, chat(model), workers=1)


def test_a_requirement_must_quote_the_posting_word_for_word(fake, job_text):
    invented = {**KUBERNETES, "quote": "You know Kubernetes inside out and love YAML."}
    answers = iter([invented, KUBERNETES])
    fake.chat_handler = lambda body: reply(
        body,
        json.dumps({"position": "ML Engineer", "company": "", "requirements": [next(answers)]}),
    )

    job = read_job(job_text, chat(fake))

    assert job.requirements[0].quote == KUBERNETES["quote"]
    assert "quote is not in the posting" in fake.requests[1][1]["messages"][-1]["content"]


def test_the_agent_asks_the_candidate_and_keeps_their_words_not_its_own(cv, model, job_text):
    job, matches = prepare(cv, model, job_text)
    said = "Yes, I ran a small k3s cluster at home for two years."
    questions = []

    findings = find_gaps(cv, job, matches, chat(model), lambda q: questions.append(q) or said)

    (kubernetes,) = [f for f in findings if f.status == "covered"]
    assert kubernetes.fact == said and kubernetes.bullet_ids == []
    assert questions == ["Have you used Kubernetes in production?"]
    assert len(findings) == 2  # Kubernetes, and the time-series line nothing in the CV shows


def test_a_requirement_the_candidate_cannot_answer_stays_a_gap(cv, model, job_text):
    job, matches = prepare(cv, model, job_text)

    findings = find_gaps(cv, job, matches, chat(model), lambda q: "")

    assert {f.status for f in findings} == {"gap"}


def test_a_reworded_bullet_cannot_bring_a_name_its_source_lacks(cv, fake):
    job = read_job("Experience with PyTorch in production.", chat(_posting(fake)))
    sent = []

    def handler(body):
        sent.append(body["messages"][-1]["content"])
        text = "Trained transformer models in PyTorch for text classification"
        if len(sent) == 1:
            text += " on Kubernetes"  # the source bullet never mentions it
        bullets = [{"source_id": "r1.1", "text": text}]
        return reply(body, json.dumps({"role_id": "x", "bullets": bullets}))

    fake.chat_handler = handler
    first_role = cv.model_copy(update={"roles": cv.roles[:1]})

    (role,) = tailor_roles(first_role, job, chat(fake), workers=1)

    assert "Kubernetes" not in role.bullets[0].text
    assert "kubernetes" in sent[1].lower()  # the model was told which name had no source


def _posting(fake):
    fake.chat_handler = lambda body: reply(
        body,
        json.dumps({"position": "ML Engineer", "company": "", "requirements": [
            {"text": "PyTorch", "quote": "Experience with PyTorch in production.", "must": True}
        ]}),
    )  # fmt: skip
    return fake


def test_the_letter_names_at_most_four_experiences_and_counts_years_from_the_cv(
    cv, model, job_text
):
    job, matches = prepare(cv, model, job_text)

    highlights = write_highlights(cv, job, matches, [], chat(model))

    assert 1 <= len(highlights) <= 4
    pytorch = Highlight(requirement_id="q1", skill="PyTorch", reason="I trained models in PyTorch.")
    assert years_sentence(cv, [pytorch], date(2026, 10, 3)) == (
        "I have 7 years of experience in PyTorch."  # PyTorch since 2019 in the CV
    )
    unknown = Highlight(requirement_id="q1", skill="Go", reason="I wrote Go services.")
    assert years_sentence(cv, [unknown], date(2026, 10, 3)) == ""  # no start year, no claim


def test_three_pdfs_come_out_and_the_gap_report_says_what_the_cv_lacks(
    model, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(f'[cache]\npath = "{tmp_path / "cache.sqlite"}"\n')
    out = tmp_path / "out"
    argv = ["cv2job", str(EXAMPLES / "cv.yaml"), str(EXAMPLES / "job.txt"), "--out", str(out)]

    assert main(argv, ollama=model.transport(), ask=lambda question: "") == 0

    letter = pdf_text(out / "cover_letter.pdf")
    cv_pdf = pdf_text(out / "cv.pdf")
    gaps = pdf_text(out / "gap_report.pdf")
    assert letter.startswith("Dear Hiring Manager,") and "Contoso Vision" in letter
    assert "Kubernetes" not in cv_pdf and "Kubernetes" not in letter
    not_covered = gaps.split("NOT COVERED")[1].split(" COVERED ")[0]
    assert "Kubernetes" in not_covered and "PyTorch" not in not_covered
