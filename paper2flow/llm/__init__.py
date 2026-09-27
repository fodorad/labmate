"""Model access: Ollama client, request/response types and the record/replay cache."""

from paper2flow.llm.client import Backend, OllamaClient, OllamaError
from paper2flow.llm.replay import CassetteMissError, CassetteStore, ReplayClient
from paper2flow.llm.structured import extract_json, parse_structured
from paper2flow.llm.types import (
    ChatRequest,
    ChatResponse,
    ImageRequest,
    ImageResponse,
    Message,
    ToolCall,
    ToolFunction,
    Usage,
)

__all__ = [
    "Backend",
    "CassetteMissError",
    "CassetteStore",
    "ChatRequest",
    "ChatResponse",
    "ImageRequest",
    "ImageResponse",
    "Message",
    "OllamaClient",
    "OllamaError",
    "ReplayClient",
    "ToolCall",
    "ToolFunction",
    "Usage",
    "extract_json",
    "parse_structured",
]
