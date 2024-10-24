from typing import Any, AsyncGenerator, Coroutine, Dict, Generator, List, Mapping, Optional, Tuple
from httpx import URL, Client, Timeout
from pydantic import model_validator, SecretStr, Field, BaseModel
from prompt_scope.core.schemas.message import ChatMessage, ChatResponse, MessageRole
from prompt_scope.core.llms.base import BaseLLM


class HuggingFaceLLM(BaseLLM):
    llm: Any
    tokenizer: Any = None
    model_id: Optional[str] = None

    @model_validator
    @classmethod
    def validate_client(cls, data: Any):
        return data

    def chat(self, messages: List[ChatMessage] | str, **kwargs) -> ChatResponse:
        return super().chat(messages, **kwargs)

    def structured_output(self, messages: List[ChatMessage], method: str, schema: BaseModel | Dict) -> ChatResponse:
        return super().structured_output(messages, method, schema)
