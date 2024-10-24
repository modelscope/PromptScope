from typing import Type

from pydantic import BaseModel, ValidationError

from prompt_scope.core.output_parsers.json import JsonOutputParser
from loguru import logger


class PydanticOutputParser(JsonOutputParser):

    def __init__(self, schema: Type[BaseModel]):
        if not issubclass(schema, BaseModel):
            raise ValueError("schema must be a subclass of BaseModel")
        self.schema = schema
        self.logger = logger

    def parse(self, text: str | dict, **kwargs) -> BaseModel:
        try:
            if isinstance(text, dict):
                return self.schema.model_validate(text)

            content = super().parse(text)
            return self.schema.model_validate(content)
        except ValidationError as e:
            self.logger.warning(f"Validation Error: {e} for input text {text}")
            raise ValueError(f"Pydantic object can not be parsed, origin input {text}")
