
from __future__ import annotations

import json
import re
from typing import Any, Callable


_json_markdown_re = re.compile(r"```(json)?(.*)", re.DOTALL)
def parse_json_markdown(
    json_string: str, *, parser: Callable[[str], Any] = json.loads
) -> dict:
    """Parse a JSON string from a Markdown string.

    Args:
        json_string: The Markdown string.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    try:
        return _parse_json(json_string, parser=parser)
    except json.JSONDecodeError:
        # Try to find JSON string within triple backticks
        match = _json_markdown_re.search(json_string)

        # If no match found, assume the entire string is a JSON string
        # Else, use the content within the backticks
        json_str = json_string if match is None else match.group(2)
    return _parse_json(json_str, parser=parser)


_json_strip_chars = " \n\r\t`"
def _parse_json(
    json_str: str, *, parser: Callable[[str], Any] = json.loads
) -> dict:
    # Strip whitespace,newlines,backtick from the start and end
    json_str = json_str.strip(_json_strip_chars)

    # Parse the JSON string into a Python dictionary
    return parser(json_str)