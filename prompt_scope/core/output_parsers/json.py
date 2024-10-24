import json
from typing import Dict, Any

from prompt_scope.core.output_parsers.base import BaseOutputParser
from prompt_scope.core.utils.json import parse_partial_json
from loguru import logger


# some bug here when parse json
class JsonOutputParser(BaseOutputParser[Dict[str, Any]]):

    def parse(self, text: str, **kwargs) -> Dict[str, Any]:
        """Parse the output of an LLM call."""
        content = dict()
        try:
            content = parse_partial_json(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Json decode error {e} for text: {text}")
        except Exception as e:
            logger.warning(f"Unexpected error: {e} for text: {text}")

        return content
