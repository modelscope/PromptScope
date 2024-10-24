from typing import Any

from prompt_scope.core.output_parsers.base import BaseOutputParser


class DummyOpenaiParser(BaseOutputParser):
    def parse(self, text: str, **kwargs) -> Any:
        return text
