import json
from typing import Generator, Any, Union

from pydantic import BaseModel, Field

from prompt_scope.core.enumeration.model_enum import ModelEnum
from prompt_scope.core.scheme.message import Message


class ModelResponse(BaseModel):
    message: Union[Message, None] = Field(None, description="generation model result")

    delta: str = Field("", description="New text that just streamed in (only used when streaming)")

    m_type: ModelEnum = Field(ModelEnum.GENERATION_MODEL, description="One of LLM, EMB, RANK.")

    status: bool = Field(True, description="Indicates whether the model call was successful.")

    details: str = Field("", description="The details information for model call, "
                                         "usually for storage of raw response or failure messages.")

    raw: Any = Field("", description="Raw response from model call")

    def __str__(self, max_size=100, **kwargs):
        result = {}
        for key, value in self.model_dump().items():
            if key == "raw" or not value:
                continue

            if isinstance(value, str):
                result[key] = value
            elif isinstance(value, (list, dict)):
                result[key] = f"{str(value)[:max_size]}... size={len(value)}"
            elif isinstance(value, ModelEnum):
                result[key] = value.value
        return json.dumps(result, **kwargs)


ModelResponseGen = Generator[ModelResponse, None, None]

# class QwenResponse(BaseModel):
#     status_code: int = Field(200, description="model response status code")

#     request_id: str = Field("", description="model request id")

#     output: dict = Field({}, description="contain the model response")

#     usage: dict = Field({}, description="tokens used")
