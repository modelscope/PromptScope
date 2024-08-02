from typing import List, Union, Optional, Any
import os

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, CompletionResponse
from llama_index.llms.dashscope import DashScope
from dashscope.aigc.generation import AioGeneration

from meta_icl.core.enumeration.message_role_enum import MessageRoleEnum
from meta_icl.core.enumeration.model_enum import ModelEnum
from meta_icl.core.models.base_model import BaseModel, MODEL_REGISTRY
from meta_icl.core.scheme.message import Message
from meta_icl.core.scheme.model_response import ModelResponse, ModelResponseGen
# import dashscope
import asyncio

os.environ['DASHSCOPE_API_KEY'] = ''

class LlamaIndexGenerationModel(BaseModel):
    m_type: ModelEnum = ModelEnum.GENERATION_MODEL
    MODEL_REGISTRY.register("dashscope_generation", DashScope)

    def before_call(self, **kwargs):
        prompt: Union[List[str], str] = kwargs.pop("prompt", "")
        messages: Union[Union[List[Message], List[dict]], List[Union[List[Message], List[dict]]]] = kwargs.pop("messages", [])
        if prompt:
            if isinstance(prompt, List):
                self.data = [{"prompt": p} for p in prompt]
            elif isinstance(prompt, str):
                self.data = {"prompt": prompt}
        elif messages:
            if isinstance(messages, List[Union[List[Message], List[dict]]]):
                if isinstance(messages[0][0], dict):
                    self.data = [{"messages": [ChatMessage(role=m["role"], content=m["content"]) for m in msg]} for msg in messages]
                else:
                    self.data = [{"messages": [ChatMessage(role=m.role, content=m.content) for m in msg]} for msg in messages]
            elif isinstance(messages, Union[List[Message], List[dict]]):
                if isinstance(messages[0], dict):
                    self.data = {"messages": [ChatMessage(role=msg["role"], content=msg["content"]) for msg in messages]}
                else:
                    self.data = {"messages": [ChatMessage(role=msg.role, content=msg.content) for msg in messages]}
        else:
            raise RuntimeError("prompt and messages are both empty!")

    def after_call(self,
                   model_response: Union[ModelResponse, List[ModelResponse]],
                   stream: bool = False,
                   **kwargs) -> Union[ModelResponse, ModelResponseGen]:
        model_response.message = Message(role=MessageRoleEnum.ASSISTANT, content="")

        call_result = model_response.raw
        if stream:
            def gen() -> ModelResponseGen:
                for response in call_result:
                    model_response.message.content += response.delta
                    model_response.delta = response.delta
                    yield model_response
            return gen()
        else:
            if isinstance(call_result, CompletionResponse):
                model_response.message.content = call_result.text
            elif isinstance(call_result, ChatResponse):
                model_response.message.content = call_result.message.content
            else:
                raise NotImplementedError

            return model_response

    def _call(self, stream: bool = False, **kwargs) -> Union[ModelResponse, ModelResponseGen]:
        assert "prompt" in self.data or "messages" in self.data
        results = ModelResponse(m_type=self.m_type)

        if "prompt" in self.data:
            if stream:
                response = self.model.stream_complete(**self.data)
            else:
                response = self.model.complete(**self.data)
        else:
            if stream:
                response = self.model.stream_chat(**self.data)
            else:
                response = self.model.chat(**self.data)
        results.raw = response
        return results

    async def _async_call(self, data: dict, **kwargs):
        assert "prompt" in data or "messages" in data
        results = ModelResponse(m_type=self.m_type)
        if "prompt" in data:
            results.raw = await self.model.acomplete(**data)
        elif "messages" in data:
            results.raw = await self.model.achat(**data)
        else:
            raise RuntimeError("prompt or messages is missing!")
        return results

class AioGenerationWithInit(AioGeneration):
    def __init__(
            self,
            model_name: Optional[str] = "qwen2",
            max_tokens: Optional[int] = 2000,
            temperature: Optional[float] = 0.85,
            seed: Optional[int] = 1234,
            api_key: Optional[str] = None,
            **kwargs: Any,
        ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.api_key = api_key

class AioGenerationModel(BaseModel):
    m_type: ModelEnum = ModelEnum.GENERATION_MODEL
    MODEL_REGISTRY.register("aio_generation", AioGenerationWithInit)

    def before_call(self, **kwargs):
        self.prompts: List = kwargs.pop("prompt", [])
        self.messages: Union[List[Message], List[dict]] = kwargs.pop("messages", [])
        if not (self.prompts or self.messages):
            raise RuntimeError("prompt and messages are both empty!")
        

    def after_call(self,
                   model_response: Union[ModelResponse, List[ModelResponse]],
                   stream: bool = False,
                   **kwargs) -> Union[ModelResponse, ModelResponseGen]:
        pass # No-op implementation

    def _call():
        pass # No-op implementation

    async def _async_call(self, prompt: str="", messages: List=[], **kwargs):
        if prompt:
            return await self.model.call(model=self.model_name, prompt=prompt)
        elif messages:
            return await self.model.call(model=self.model_name, messages=messages)