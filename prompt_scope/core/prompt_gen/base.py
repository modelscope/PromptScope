import os
from abc import ABC
from typing import List, Any
from pydantic import BaseModel
from dataclasses import field

from ..utils.utils import load_yaml
from ..schemas.example import Example


class LLMConfig(BaseModel, ABC):
    api_type: str = field(metadata={"allowed_values": {"dashscope", "openai"}})
    client: str
    model: str
    api_key: str = None
    top_p: float = None
    temperature: float = None
    presence_penalty: float = None
    max_tokens: float = None
    seed: float = None

    @staticmethod
    def load(load_path: str):
        llm_config = load_yaml(load_path)
        if "api_key" not in llm_config or llm_config["api_key"] is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if api_key is not None:
                llm_config["api_key"] = api_key
            else:
                raise RuntimeError("Dashscope API key not found")

        return LLMConfig(**llm_config)


class Tool(BaseModel):
    type: str
    function: Any


class PromptParams(BaseModel):
    system_prompt: str | None
    user_prompt: str
    examples: List[Example] | None = None
    toos: List[Tool] = None
    placeholders: List[str] = []






