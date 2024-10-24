import re
from typing import List

from prompt_scope.core.output_parsers.base import BaseOutputParser
from loguru import logger


class RegexOutputParser(BaseOutputParser[List[str]]):
    def __init__(self, regex_pattern: str):
        if not regex_pattern:
            raise ValueError("regex_pattern cannot be empty.")
        self.regex_pattern = re.compile(regex_pattern)

    def parse(self, text: str, **kwargs) -> List[str]:
        """Parse the output of an LLM call to return all matches."""
        matches = self.regex_pattern.findall(text)
        if matches:
            return matches
        Logger.get_logger().warning(f"No match found for pattern {self.regex_pattern.pattern} in text: {text}")
        return []
