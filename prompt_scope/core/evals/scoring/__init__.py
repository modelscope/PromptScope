from pydantic import Field, BaseModel

class ScoreSchema(BaseModel):
    comment: str = Field(..., description="Comments for the answer and the score")
    score: str = Field(..., description="The score as a string")