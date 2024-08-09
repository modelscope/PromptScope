from typing import List, Union, Optional, Any
import os

import dashscope
from dashscope.aigc.generation import AioGeneration
from dashscope import Generation
from openai import OpenAI, AsyncOpenAI

from meta_icl.core.enumeration.message_role_enum import MessageRoleEnum
from meta_icl.core.enumeration.model_enum import ModelEnum
from .base_model import BaseModel, BaseAsyncModel, MODEL_REGISTRY
from meta_icl.core.scheme.message import QwenMessage
from meta_icl.core.scheme.model_response import QwenResponse
# import dashscope
import asyncio

class GenerationModel(BaseModel):
    m_type: ModelEnum = ModelEnum.GENERATION_MODEL
    MODEL_REGISTRY.register("dashscope_generation", Generation)

    def _call(self, stream: bool = False, prompt: str = "", messages: List[QwenMessage] = [], **kwargs) -> QwenResponse:
        return self.call_module.call(stream=stream, model=self.model_name, prompt=prompt, messages=messages, **kwargs)
        
class AioGenerationModel(BaseAsyncModel):
    m_type: ModelEnum = ModelEnum.GENERATION_MODEL
    MODEL_REGISTRY.register("aio_generation", AioGeneration)
    async def _async_call(self, prompt: str = "", messages: List[QwenMessage] = [], **kwargs) -> QwenResponse:
        return await self.call_module.call(model=self.model_name, prompt=prompt, messages=messages, **kwargs)
    
class OpenAIGenerationModel(BaseModel):
    m_type: ModelEnum = ModelEnum.GENERATION_MODEL
    MODEL_REGISTRY.register("openai_generation", OpenAI)
    def __wrap_message(self, prompt, sys_prompt="You are a helpful assistant"):
        return [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
                ]
    def _call(self, stream: bool = False, prompt: str = "", messages: List[QwenMessage] = [], **kwargs) -> QwenResponse:
        if prompt:
            messages = self.__wrap_message(prompt)
        return self.call_module.chat.completions.create(stream=stream, model=self.model_name, messages=messages, **kwargs)
    
class OpenAIAioGenerationModel(BaseAsyncModel):
    m_type: ModelEnum = ModelEnum.GENERATION_MODEL
    MODEL_REGISTRY.register("openai_aio_generation", AsyncOpenAI)
    def __wrap_message(self, prompt, sys_prompt="You are a helpful assistant"):
        return [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
                ]
    async def _async_call(self, prompt: str = "", messages: List[QwenMessage] = [], **kwargs) -> QwenResponse:
        if prompt:
            messages = self.__wrap_message(prompt)
        return await self.call_module.chat.completions.create(model=self.model_name, messages=messages, **kwargs)