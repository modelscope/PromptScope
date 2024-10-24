import inspect
from typing import Any, Dict, List, Type, Optional, Union
from enum import Enum

import openai
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from pydantic import model_validator, SecretStr, Field, BaseModel

from prompt_scope.core.llms.base import BaseLLM
from prompt_scope.core.output_parsers.pydantic import PydanticOutputParser
from prompt_scope.core.schemas.message import ChatMessage, ChatResponse, ChatTool
from prompt_scope.core.utils.env import get_from_dict_or_env


class OpenaiLlmModel(str, Enum):
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    O1_PREVIEW = "o1-preview"
    O1_MINI = "o1-mini"

def _convert_completion_to_response(completion: ChatCompletion | ChatCompletionChunk) -> ChatResponse:
    """
    Convert ChatCompletion to ChatResponse
    """
    if isinstance(completion, ChatCompletion):
        message = completion.choices[0].message
    elif isinstance(completion, ChatCompletionChunk):
        message = completion.choices[0].delta
    else:
        raise TypeError(f"Unsupported type: {type(completion)}")

    message = ChatMessage(
        role=message.role if message.role is not None else "assistant",
        name=message.name if hasattr(message, "name") else None,
        tool_calls=[ChatTool(**tool.model_dump()) for tool in message.tool_calls] if message.tool_calls else None,
        content=message.content,
        additional_kwargs=message.model_dump(exclude={"role", "content", "tool_calls"})
    )
    return ChatResponse(
        message=message,
        raw=completion.model_dump(),
        additional_kwargs={}
    )


def _convert_to_openai_message(messages: List[ChatMessage]):
    return [
        {
            "role": message.role.name.lower(),
            "content": message.content
        }
        for message in messages
    ]


class OpenaiLLM(BaseLLM):
    client: Any
    async_client: Any
    open_api_key: SecretStr = Field(default=None, alias="api_key")
    openai_api_base: str = Field(default=None, alias="base_url")
    organization: str = Field(default=None)
    project: str = Field(default=None)
    base_url: str = Field(default=None)
    timeout: float = Field(default=None)
    max_retries: int = Field(default=2)
    default_headers: Union[Dict[str, str], None] = Field(default=None)
    default_query: Union[Dict[str, Any], None] = Field(default=None)
    http_client: Optional[Any] = Field(default=None)
    strict_response_validation: bool = Field(default=False)
    model: str = Field(default=OpenaiLlmModel.GPT_4O)
    parallel_tool_calls: bool = Field(default=True, description="Whether to use parallel tool calls")

    @model_validator(mode="before")
    @classmethod
    def validate_client(cls, data: Dict) -> Dict:
        """
        Create openai client from api key and base url.
        """
        data["openai_api_key"] = get_from_dict_or_env(data=data, key="openai_api_key")
        data["openai_api_base"] = get_from_dict_or_env(data=data, key="openai_api_base")

        client_kwargs = {
            "api_key": data.get("openai_api_key", None),
            "organization": data.get("organization", None),
            "project": data.get("project", None),
            "base_url": data.get("openai_api_base", None),
            "timeout": data.get("timeout", None),
            "max_retries": data.get("max_retries", None),
            "default_headers": data.get("default_headers", None),
            "default_query": data.get("default_query", None),
            "http_client": data.get("http_client", None),
            "_strict_response_validation": data.get("strict_response_validation", None)
        }

        data["client"] = openai.OpenAI(**{k: v for k, v in client_kwargs.items() if v is not None})
        data["async_client"] = openai.AsyncOpenAI(**{k: v for k, v in client_kwargs.items() if v is not None})
        return data

    @property
    def chat_kwargs(self) -> Dict[str, Any]:
        call_params = {
            "model": self.model,
            "top_p": self.top_p,
            "temperature": self.temperature,
        }
        if self.tools:
            call_params.update({
                "tools": self.tools,
                "tool_choice": self.tool_choice,
                "parallel_tool_calls": self.parallel_tool_calls
            })
        return call_params

    def chat(self, messages: Union[List[ChatMessage], str], **kwargs) -> ChatResponse:
        messages = self._convert_messages(messages)  # type: ignore

        # update the chat kwargs according to the kwargs
        call_params = self.chat_kwargs
        call_params.update(kwargs)

        if "response_format" in kwargs and inspect.isclass(kwargs["response_format"]):
            response = self.client.beta.chat.completions.parse(
                messages=_convert_to_openai_message(messages),
                **call_params
            )
        else:
            response = self.client.chat.completions.create(
                messages=_convert_to_openai_message(messages),
                **call_params
            )
        response = _convert_completion_to_response(response)
        return response

    def structured_output(
            self,
            messages: List[ChatMessage] | str,
            schema: Type[BaseModel],
            method: str = "json_schema",
            strict: Optional[bool] = None,
            **kwargs
    ) -> BaseModel:
        if method == "base":
            return super().structured_output(messages, schema, **kwargs)

        messages = self._convert_messages(messages)
        chat_kwargs = dict(messages=messages)
        output_parser = PydanticOutputParser(schema)
        if method == "function_calling":
            tool = openai.pydantic_function_tool(schema)
            chat_kwargs["tools"] = [tool]
            chat_kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": tool["function"]["name"]}
            }
        elif method == "json_mode":
            chat_kwargs["response_format"] = {"type": "json_object"}
        elif method == "json_schema":
            chat_kwargs["response_format"] = schema
        else:
            raise ValueError(
                f"Unrecognized method argument. "
                f"Expected one of 'base', 'function_calling', 'json_schema' or 'json_mode'. Received: '{method}'"
            )

        response = self.chat(**chat_kwargs, **kwargs)
        if method == "function_calling":
            parsed = output_parser.parse(text=response.message.tool_calls[0].function.arguments)
        elif method == "json_mode":
            parsed = output_parser(response)
        else:
            parsed = output_parser(response)

        return parsed
