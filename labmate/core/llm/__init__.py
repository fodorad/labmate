"""Model access: Ollama client, request/response types and the record/replay cache."""

from labmate.core.llm.client import Backend, OllamaClient, OllamaError
from labmate.core.llm.replay import CassetteMissError, CassetteStore, ReplayClient
from labmate.core.llm.structured import extract_json, parse_structured
from labmate.core.llm.types import (
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
