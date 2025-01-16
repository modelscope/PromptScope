"""Base LLM abstract class.
"""

import asyncio
from typing import List, Dict, Any, Union, Optional, Type
from loguru import logger
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

from prompt_scope.core.output_parsers.llm_parser import LlmOutputParser, PYDANTIC_FORMAT_INSTRUCTIONS, \
    PYDANTIC_FORMAT_INSTRUCTIONS_SIMPLE
from prompt_scope.core.schemas.message import ChatMessage, ChatResponse, MessageRole


class BaseLLM(BaseModel):
    model: str
    temperature: float = 0.85
    top_p: float = 1.0
    top_k: Optional[int] = None
    max_tokens: int = Field(default=2048, description="Max tokens to generate for llm.")
    stop: List[str] = Field(default_factory=list, description="List of stop words")
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of tools to use")
    tool_choice: Union[str, Dict] = Field(default="auto", description="tool choice when user passed the tool list")
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    @staticmethod
    def _convert_messages(messages: List[ChatMessage] | ChatMessage | str) -> List[ChatMessage]:
        if isinstance(messages, list):
            return messages
        elif isinstance(messages, str):
            return [ChatMessage(content=messages, role=MessageRole.USER)]
        elif isinstance(messages, ChatMessage):
            assert messages.role == MessageRole.USER, "Only support user message."
            return [messages]
        else:
            raise ValueError(
                f"Invalid message type {messages}. "
            )

    def chat(self, messages: List[ChatMessage] | str, **kwargs) -> ChatResponse:
        """

        Args:
            messages:
            **kwargs:

        Returns:

        """

        raise NotImplementedError

    async def achat(self, index, messages: Union[List[ChatMessage], str], **kwargs) -> ChatResponse:
        """

        Args:
            messages:
            **kwargs:

        Returns:

        """
        if kwargs.get("semaphore", 0) > 0:
            _semaphore = asyncio.Semaphore(kwargs.get("semaphore", 0))
            async with _semaphore:
                result = await asyncio.to_thread(self.chat, messages, **kwargs)
                return {"index": index, "result": result}
        else:
            result = await asyncio.to_thread(self.chat, messages, **kwargs)
            return {"index": index, "result": result}

    def structured_output(
            self,
            schema: Type[BaseModel],
            messages: Union[List[ChatMessage], str] = [],
            example_instruction: bool = False,
            **kwargs
    ) -> BaseModel:
        output_parser = LlmOutputParser(
                llm=self, schema=schema,
                format_prompt_template=PYDANTIC_FORMAT_INSTRUCTIONS
                if example_instruction else PYDANTIC_FORMAT_INSTRUCTIONS_SIMPLE
            )
        messages = self._convert_messages(messages)
        messages = output_parser.generate_prompt(messages=self._convert_messages(messages))
        response = self.chat(messages, **kwargs)
        parsed = output_parser.parse(response.message.content)
        return parsed
    
    async def astructured_output(
            self,
            schema: Type[BaseModel],
            list_of_messages: List[Union[List[ChatMessage], str]] = [],
            example_instruction: bool = False,
            **kwargs
    ) -> BaseModel:
        output_parser = LlmOutputParser(
                llm=self, schema=schema,
                format_prompt_template=PYDANTIC_FORMAT_INSTRUCTIONS
                if example_instruction else PYDANTIC_FORMAT_INSTRUCTIONS_SIMPLE
            )
        list_of_messages = list(map(self._convert_messages, list_of_messages))
        list_of_messages = list(map(output_parser.generate_prompt, list_of_messages))
        tasks = [self.achat(i, messages, **kwargs) for i, messages in enumerate(list_of_messages)]
        responses = []
        for result in tqdm.as_completed(tasks):
            # As each task completes, the progress bar will be updated
            value = await result
            responses.append(value)
        responses = [response["result"] for response in sorted(responses, key=lambda x: x["index"])]
        parsed = list(map(lambda resp: output_parser.parse(resp.message.content), responses))
        return parsed

    def simple_chat(self, query: str, history: Optional[List[str]] = None, sys_prompt: str = "") -> Any:
        messages = [ChatMessage(role=MessageRole.SYSTEM, content=sys_prompt)]

        if history is None:
            history = []
        history += [query]

        for i, h in enumerate(history):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            messages += [ChatMessage(role=role, content=h)]

        response: ChatResponse = self.chat(messages)

        return response.message.content
