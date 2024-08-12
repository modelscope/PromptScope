from typing import List, Union, Optional, Any, cast
import os

import dashscope
from dashscope.aigc.generation import AioGeneration
from dashscope import Generation
from openai import OpenAI, AsyncOpenAI
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from meta_icl.core.enumeration.message_role_enum import MessageRoleEnum
from meta_icl.core.enumeration.model_enum import ModelEnum
from meta_icl.core.models.base_model import BaseModel, BaseAsyncModel, MODEL_REGISTRY
from meta_icl.core.scheme.message import Message
from meta_icl.core.scheme.model_response import ModelResponse, ModelResponseGen
# import dashscope
import asyncio

class GenerationModel(BaseModel):
    m_type: ModelEnum = ModelEnum.GENERATION_MODEL
    MODEL_REGISTRY.register("dashscope_generation", Generation)

    def _call(self, stream: bool = False, prompt: str = "", messages: List[Message] = [], **kwargs) -> ModelResponse:
        return self.call_module.call(stream=stream, model=self.model_name, prompt=prompt, messages=messages, **kwargs)

    def after_call(self,
                   model_response: ModelResponse,
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
            model_response.message.content = call_result.output.text

        return model_response
        
class AioGenerationModel(BaseAsyncModel):
    m_type: ModelEnum = ModelEnum.GENERATION_MODEL
    MODEL_REGISTRY.register("aio_generation", AioGeneration)
    async def _async_call(self, prompt: str = "", messages: List[Message] = [], **kwargs) -> ModelResponse:
        return await self.call_module.call(model=self.model_name, prompt=prompt, messages=messages, **kwargs)
    
    def after_call(self,
                   model_response: ModelResponse,
                   **kwargs) -> ModelResponse:
        model_response.message = Message(role=MessageRoleEnum.ASSISTANT, content="")

        call_result = model_response.raw
        model_response.message.content = call_result.output.text

        return model_response
    
class OpenAIGenerationModel(BaseModel):
    m_type: ModelEnum = ModelEnum.GENERATION_MODEL
    MODEL_REGISTRY.register("openai_generation", OpenAI)
    def __wrap_message(self, prompt, sys_prompt="You are a helpful assistant"):
        return [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
                ]
    def _call(self, stream: bool = False, prompt: str = "", messages: List[Message] = [], **kwargs) -> ModelResponse:
        if prompt:
            messages = self.__wrap_message(prompt)
        return self.call_module.chat.completions.create(stream=stream, model=self.model_name, messages=messages, **kwargs)
    
    def after_call(self,
                   model_response: ModelResponse,
                   stream: bool = False,
                   **kwargs) -> Union[ModelResponse, ModelResponseGen]:
        model_response.message = Message(role=MessageRoleEnum.ASSISTANT, content="")

        call_result = model_response.raw
        if stream:
            def gen() -> ModelResponseGen:
                for response in call_result:
                    response = cast(ChatCompletionChunk, response)
                    model_response.message.content += response.choices[0].delta.content or ""
                    model_response.delta = response.choices[0].delta
                    yield model_response

            return gen()
        else:
            model_response.message.content = call_result.choices[0].message.content

        return model_response
    
class OpenAIAioGenerationModel(BaseAsyncModel):
    m_type: ModelEnum = ModelEnum.GENERATION_MODEL
    MODEL_REGISTRY.register("openai_aio_generation", AsyncOpenAI)
    def __wrap_message(self, prompt, sys_prompt="You are a helpful assistant"):
        return [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
                ]
    async def _async_call(self, prompt: str = "", messages: List[Message] = [], **kwargs) -> ModelResponse:
        if prompt:
            messages = self.__wrap_message(prompt)
        return await self.call_module.chat.completions.create(model=self.model_name, messages=messages, **kwargs)
    
    def after_call(self,
                   model_response: ModelResponse,
                   **kwargs) -> ModelResponse:
        model_response.message = Message(role=MessageRoleEnum.ASSISTANT, content="")

        call_result = model_response.raw
        model_response.message.content = call_result.choices[0].message.content

        return model_response