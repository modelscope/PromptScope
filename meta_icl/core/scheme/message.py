import datetime
from typing import Dict

from pydantic import Field, BaseModel


class Message(BaseModel):
    role: str = Field(..., description="The role of the message sender (user, assistant, system)")

    role_name: str = Field("", description="role name")

    content: str = Field(..., description="The body of the message")

    time_created: int = Field(int(datetime.datetime.now().timestamp()),
                              description="Timestamp when the message was created")

    meta_data: Dict[str, str] = Field({}, description="meta data for msg")

class QwenMessage(BaseModel):
    role: str = Field(..., description="The role of the message sender (user, assistant, system)")

    content: str = Field(..., description="The body of the message")   