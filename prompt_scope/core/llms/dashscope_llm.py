from typing import Any, Dict, List, Optional, Type, Union
from enum import Enum
import asyncio
from tqdm.asyncio import tqdm
import dashscope
import openai
from dashscope.api_entities.dashscope_response import GenerationResponse, Message
from pydantic import model_validator, Field, BaseModel

from prompt_scope.core.llms.base import BaseLLM
from prompt_scope.core.output_parsers.pydantic import PydanticOutputParser
from prompt_scope.core.schemas.message import ChatMessage, ChatResponse
from prompt_scope.core.utils.env import get_from_dict_or_env


class DashScopeLlmName(str, Enum):
    QWEN_MAX = "qwen-max"
    QWEN_PLUS = "qwen-plus"
    QWEN_TURBO = "qwen-turbo"
    QWEN_LONG = "qwen-long"
    QWEN2_7B_INST = "qwen2-7b-instruct"
    QWEN2_72B_INST = "qwen2-72b-instruct"


def _convert_chat_message_to_dashscope_message(messages: List[ChatMessage]) -> List[Message]:
    return [
        Message(
            role=message.role.name.lower(),
            content=message.content,  # type: ignore
        ) for message in messages
    ]


def _convert_dashscope_response_to_response(response: GenerationResponse) -> ChatResponse:
    message = response.output.choices[0].message
    additional_kwargs = {}  # TODO
    message = ChatMessage(
        role=message.role,  # type: ignore
        content=message.content,
        name=message.get("name", None),
        tool_calls=message.get("tool_calls", None),
        additional_kwargs=additional_kwargs
    )

    return ChatResponse(
        message=message,
        raw=response
    )


class DashscopeLLM(BaseLLM):
    client: Any
    async_client: Any
    model: str = Field(default=DashScopeLlmName.QWEN2_7B_INST)
    dashscope_api_key: Optional[str] = Field(default=None)
    max_retries: int = Field(default=10)

    @model_validator(mode="before")
    @classmethod
    def validate_client(cls, data: Dict):
        """
        Create a DashScope client from the provided API key.
        """
        data["dashscope_api_key"] = get_from_dict_or_env(data=data, key="dashscope_api_key")
        data["async_client"] = dashscope.aigc.generation.AioGeneration
        data["client"] = dashscope.Generation
        return data

    @property
    def chat_kwargs(self) -> Dict[str, Any]:
        call_params = {
            "model": self.model,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key": self.dashscope_api_key,
            "result_format": "message",
        }
        if self.tools:
            call_params.update({
                "tools": self.tools,
                "tool_choice": self.tool_choice
            })

        return call_params

    def chat(self, messages: List[ChatMessage] | str, **kwargs) -> ChatResponse:
        messages = self._convert_messages(messages)
        # update the chat kwargs according to the kwargs
        call_params = self.chat_kwargs
        call_params.update(kwargs)
        response = self.client.call(
            messages=_convert_chat_message_to_dashscope_message(messages),  # type: ignore
            **call_params
        )
        response = _convert_dashscope_response_to_response(response)
        return response

    def structured_output(
            self,
            schema: Type[BaseModel],
            messages: Union[List[ChatMessage], str] = [],
            method: str = "function_calling",
            **kwargs
    ) -> BaseModel:
        if method not in ("base", "function_calling"):
            raise ValueError(
                f"Unrecognized method argument. Only support 'base', 'function_calling' "
                f"for dashscope llm: '{method}'"
            )
        if method == "base":
            return super().structured_output(messages=messages, 
                                             schema=schema,
                                             **kwargs)  # type: ignore
        else:
            # Only supports non-nested pydantic BaseModel objects.
            chat_kwargs = {}
            json_schema = openai.pydantic_function_tool(schema)
            chat_kwargs["tools"] = [json_schema]
            chat_kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": json_schema["function"]["name"]}
            }
            output_parser = PydanticOutputParser(schema)
            messages = self._convert_messages(messages)
            chat_kwargs["messages"] = messages
            for _ in range(self.max_retries):
                try:
                    response = self.chat(**chat_kwargs, **kwargs)
                    parsed = output_parser.parse(
                        response.message.tool_calls[0].function.arguments)
                except:
                    continue
            return parsed
        
    async def astructured_output(
            self,
            schema: Type[BaseModel],
            list_of_messages: List[Union[List[ChatMessage], str]] = [],
            method: str = "function_calling",
            **kwargs
    ) -> BaseModel:
        if method not in ("base", "function_calling"):
            raise ValueError(
                f"Unrecognized method argument. Only support 'base', 'function_calling' "
                f"for dashscope llm: '{method}'"
            )
        if method == "base":
            return super().astructured_output(schema=schema, 
                                             list_of_messages=list_of_messages, 
                                             **kwargs)  # type: ignore
        else:
            # Only supports non-nested pydantic BaseModel objects.
            chat_kwargs = {}
            json_schema = openai.pydantic_function_tool(schema)
            chat_kwargs["tools"] = [json_schema]
            chat_kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": json_schema["function"]["name"]}
            }
            output_parser = PydanticOutputParser(schema)
            tasks = [self.achat(messages, **chat_kwargs, **kwargs) for messages in list_of_messages]
            responses = []
            for result in tqdm.as_completed(tasks):
                # As each task completes, the progress bar will be updated
                value = await result
                responses.append(value)
            parsed = list(map(lambda resp: output_parser.parse(
                resp.message.tool_calls[0].function.arguments).query, responses))
            return parsed
