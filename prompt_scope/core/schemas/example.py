from pydantic import BaseModel
from typing import List


class ConversationRound(BaseModel):
    user: str
    assistant: str

    @property
    def text(self) -> str:
        return f"User: {self.user}\nBot: {self.assistant}"


class LLMCallRecord(BaseModel):
    system_prompt: str = None
    input: str
    output: str
    prediction: str | None = None
    history: List[ConversationRound] = None
    is_good_case: bool | None = None
    score: float | None = None
    tips: str = None

class Example(BaseModel):
    query: str = None
    response: str
