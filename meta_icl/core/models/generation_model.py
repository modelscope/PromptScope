from typing import List, Union, cast

import aiohttp
# import dashscope
import requests
from dashscope import Generation
from dashscope.aigc.generation import AioGeneration
from openai import OpenAI, AsyncOpenAI
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from meta_icl.core.enumeration.message_role_enum import MessageRoleEnum
from meta_icl.core.enumeration.model_enum import ModelEnum
from meta_icl.core.models.base_model import BaseModel, BaseAsyncModel, MODEL_REGISTRY
from meta_icl.core.scheme.message import Message
from meta_icl.core.scheme.model_response import ModelResponse, ModelResponseGen


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
            if call_result.output.text:
                model_response.message.content = call_result.output.text
            else:
                model_response.message.content = call_result.output.choices[0].message.content

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
        if call_result.output.text:
            model_response.message.content = call_result.output.text
        else:
            model_response.message.content = call_result.output.choices[0].message.content

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
        return self.call_module.chat.completions.create(stream=stream, model=self.model_name, messages=messages,
                                                        **kwargs)

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


class OpenAIPostModel(BaseModel):
    m_type: ModelEnum = ModelEnum.GENERATION_MODEL
    MODEL_REGISTRY.register("openai_post", '')

    def _call(self, stream: bool = False, prompt: str = "", messages: List[Message] = [], **kwargs) -> ModelResponse:
        url = "http://47.88.8.18:8088/api/ask"
        headers = {
            "Content-Type": "application/json",
            # "Authorization": "",
            "Authorization": "",
        }
        if prompt:
            messages = [
                {
                    "role": "system",
                    "content": kwargs.get("sys_prompt", "You are a helpful assistant")
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ]
        data = {
            "model": kwargs.get("model", "gpt-4o"),
            "n": 1,
            "temperature": kwargs.get("model", 0.2),
            "top_p": 0.7,
            "frequency_penalty": 0,
            "max_tokens": 2048,
            "presence_penalty": 0,
            "messages": messages
        }
        return requests.request("POST", url, headers=headers, json=data).json()

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
            model_response.message.content = call_result["data"]["response"]["choices"][0]["message"]["content"]

        return model_response


class OpenAIAioPostModel(BaseAsyncModel):
    m_type: ModelEnum = ModelEnum.GENERATION_MODEL
    MODEL_REGISTRY.register("openai_aio_post", '')

    async def _async_call(self, prompt: str = "", messages: List[Message] = [], **kwargs) -> ModelResponse:
        url = "http://47.88.8.18:8088/api/ask"
        headers = {
            "Content-Type": "application/json",
            # "Authorization": "",
            "Authorization": "",
        }
        if prompt:
            messages = [
                {
                    "role": "system",
                    "content": kwargs.get("sys_prompt", "You are a helpful assistant")
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ]
        data = {
            "model": kwargs.get("model", "gpt-4o"),
            "n": 1,
            "temperature": kwargs.get("model", 0.2),
            "top_p": 0.7,
            "frequency_penalty": 0,
            "max_tokens": 2048,
            "presence_penalty": 0,
            "messages": messages
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                return await response.json()

    def after_call(self,
                   model_response: ModelResponse,
                   **kwargs) -> ModelResponse:
        model_response.message = Message(role=MessageRoleEnum.ASSISTANT, content="")

        call_result = model_response.raw
        model_response.message.content = call_result["data"]["response"]["choices"][0]["message"]["content"]

        return model_response
