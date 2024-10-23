import pytest

from llamakit.core.evals.loading import load_evaluator
from llamakit.core.evals.schema import EvaluatorType
from llamakit.core.evals.scoring import ScoreSchema


QUESTION = "Tell me about Alibaba Group"
ANSWER = "Alibaba Group is a Chinese technology company."
REFERENCE = "Alibaba Group is a Chinese multinational technology company specializing in \
    e-commerce, retail, Internet, and technology."
def test_score_string() -> None:
    score_string_evaluator = load_evaluator(
        evaluator=EvaluatorType.SCORE_STRING
    )
    result = score_string_evaluator.evaluate_strings(
        prediction=ANSWER, input=QUESTION
    )
    print(result)
    assert isinstance(result, ScoreSchema)

def test_labeled_score_string() -> None:
    labeled_score_string_evaluator = load_evaluator(
        evaluator=EvaluatorType.LABELED_SCORE_STRING
    )
    result = labeled_score_string_evaluator.evaluate_strings(
        prediction=ANSWER, input=QUESTION, reference=REFERENCE
    )
    print(result)
    assert isinstance(result, ScoreSchema)

if __name__ == '__main__':
    test_score_string()
    test_labeled_score_string()