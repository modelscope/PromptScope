from enum import Enum
from typing import Generator, Literal, Optional, Any, List, Callable, Dict
from uuid import uuid4
from pydantic import BaseModel, Field
from datetime import datetime


class MessageRole(str, Enum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class Tool(BaseModel):
    arguments: str
    name: str


class ChatTool(BaseModel):
    id: str
    function: Tool
    type: Literal["function"]


class ChatMessage(BaseModel):
    """Chat message."""

    role: MessageRole = Field(default=MessageRole.USER)
    name: Optional[str] = Field(default=None)
    content: Optional[Any] = Field(default="")  # support str for llm or list of dict for multi-modal model
    tool_calls: Optional[List[ChatTool]] = Field(default=None)
    additional_kwargs: dict = Field(default_factory=dict)
    time_created: datetime = Field(default_factory=datetime.now,
                                   description="Timestamp marking the message creation time")

    def __str__(self) -> str:
        return f"{self.time_created.strftime('%Y-%m-%d %H:%M:%S')} {self.role.value}: {self.content}"

    def __add__(self, other: Any) -> "ChatMessage":
        """
        concat message with other message delta.
        """
        if other is None:
            return self
        elif isinstance(other, ChatMessage):
            return self.__class__(
                role=self.role,
                name=self.name,
                content=self.content + (other.content if other.content else ""),
                tool_calls=other.tool_calls,
                additional_kwargs=other.additional_kwargs
            )
        else:
            raise TypeError(
                'unsupported operand type(s) for +: "'
                f"{self.__class__.__name__}"
                f'" and "{other.__class__.__name__}"'
            )


class ChatSession(BaseModel):
    """Chat session to maintain the dialog history automatically."""

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    histories: List[ChatMessage] = Field(default_factory=list)
    max_history_round: int = Field(default=None)
    context_window: int = Field(default=None)
    tokenizer: Optional[Callable[[str], List[str]]] = None

    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the session."""
        self.histories.append(message)

    def delete_message(self, index: int = -1) -> None:
        """Delete a message from the session."""
        self.histories.pop(index)

    @property
    def messages(self) -> List[ChatMessage]:
        """Get the messages in the session."""
        messages = self.histories
        if self.max_history_round is not None:
            messages = messages[-self.max_history_round:]

        # todo: check the context window in the future.

        return messages


class ChatResponse(BaseModel):
    message: ChatMessage
    raw: Optional[dict] = None
    delta: Optional[ChatMessage] = None
    error_message: Optional[str] = None
    additional_kwargs: dict = Field(default_factory=dict)  # other information like token usage or log probs.

    def __str__(self):
        if self.error_message:
            return f"Errors: {self.error_message}"
        else:
            return str(self.message)

    def __add__(self, other: Any) -> "ChatResponse":
        """
        concat response with other response delta.
        """
        if other is None:
            return self
        elif isinstance(other, ChatResponse):
            return self.__class__(
                message=self.message + other.message,
                raw=other.raw,
                delta=other.message,
                error_message=other.error_message,
                additional_kwargs=other.additional_kwargs
            )
        else:
            raise TypeError(
                'unsupported operand type(s) for +: "'
                f"{self.__class__.__name__}"
                f'" and "{other.__class__.__name__}"'
            )