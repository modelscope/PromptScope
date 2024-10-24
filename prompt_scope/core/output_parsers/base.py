"""Interface for LLM output parser.
The information here refers to parsers that take a text output from a model try to parse it into a more structured representation.
More and more models are supporting function (or tool) calling, which handles this automatically.
It is recommended to use function/tool calling rather than output parsing.
"""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from prompt_scope.core.schemas.message import ChatResponse

T = TypeVar("T")


class BaseOutputParser(ABC, Generic[T]):

    @abstractmethod
    def parse(self, text: str, **kwargs) -> T:
        """Parse the output of an LLM call."""

    def __call__(self, response: ChatResponse, **kwargs) -> T:
        return self.parse(response.message.content, **kwargs)
