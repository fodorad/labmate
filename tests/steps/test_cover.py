import httpx

from paper2carousel.llm.client import OllamaClient
from paper2carousel.steps.cover import COVER_SIZE, make_cover


def test_cover_is_saved_with_the_theme_palette_prompt(fake, tmp_path):
    name, status = make_cover(
        "Attention", fake.client(), "x/z-image-turbo:latest", tmp_path / "cover.png"
    )
    assert (name, status) == ("cover.png", "ok")
    body = fake.requests[0][1]
    assert (body["width"], body["height"]) == COVER_SIZE
    assert "Attention" in body["prompt"] and "No text" in body["prompt"]


def test_server_errors_are_reported_not_raised(tmp_path):
    client = OllamaClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(500, json={"error": "boom"}))
    )
    name, status = make_cover("T", client, "img", tmp_path / "cover.png")
    assert name is None and "boom" in status and not (tmp_path / "cover.png").exists()
