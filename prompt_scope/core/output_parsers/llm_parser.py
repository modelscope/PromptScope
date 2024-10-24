"""
https://github.com/langchain-ai/langchain/blob/master/docs/extras/modules/model_io/output_parsers/pydantic.ipynb

Example 1.

# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    # You can add custom validation logic easily with Pydantic.
    @validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field


Example 2.

# Here's another example, but with a compound typed field.
class Actor(BaseModel):
    name: str = Field(description="name of an actor")
    film_names: List[str] = Field(description="list of names of films they starred in")
"""
import json
from typing import List, Type

from pydantic import BaseModel

from prompt_scope.core.output_parsers.pydantic import PydanticOutputParser
from prompt_scope.core.schemas.message import ChatMessage, MessageRole
from loguru import logger

PYDANTIC_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema}
```"""

PYDANTIC_FORMAT_INSTRUCTIONS_SIMPLE = """The output should be formatted as a JSON instance that conforms to the JSON schema below.
```
{schema}
```"""


PYDANTIC_JSON_REPAIR_PROMPT = """Fix a broken json string.

The broken json string need to fix is:
```
{broken_json}
```

The error message is:
{error}

Now, fix this json string. Please directly output the fixed json result without any other information.
"""


class JsonStringError(Exception):
    ...


class LlmOutputParser(PydanticOutputParser):
    def __init__(self, llm, schema: Type[BaseModel], format_prompt_template: str):
        super().__init__(schema=schema)
        self.llm = llm
        self.format_prompt_template = format_prompt_template

    def generate_format_prompt(self) -> str:
        schema = self.schema.model_json_schema()

        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure json in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema)
        return self.format_prompt_template.format(schema=schema_str)

    def generate_prompt(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        return [ChatMessage(role="system", content=self.generate_format_prompt())] + messages

    def repair(self, text: str, error: str) -> str:
        try:
            logger.info(f'Repairing json: {text}')
            repair_prompt = PYDANTIC_JSON_REPAIR_PROMPT.format(
                broken_json=text,
                error=repr(error),
            )
            messages = self.generate_prompt(
                messages=[ChatMessage(role=MessageRole.USER, content=repair_prompt)],
            )
            response = self.llm.chat(messages=messages)
            content = response.message.content
            logger.info('Repair json success.')
        except Exception as e:
            logger.info('Repair json fail.')
            raise JsonStringError('Cannot repair json.', str(e))

        return content

    def parse(self, text: str, **kwargs) -> BaseModel:
        try:
            content = super().parse(text, **kwargs)
        except Exception as e:
            logger.error(f"Failed to parse response: {e}, try repair again.")
            text = self.repair(text, repr(e))
            content = super().parse(text, **kwargs)
        return content
