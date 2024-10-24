import pytest
import re

from prompt_scope.core.evals.loading import load_evaluator
from prompt_scope.core.evals.schema import EvaluatorType

BASESENTENCE = "The price of the shirt is 9.15"

@pytest.mark.parametrize(
    "ignore_case, ignore_punctuation, ignore_numbers, prediction, reference, score", 
    [
        (False, False, False, BASESENTENCE, BASESENTENCE, 1.0),
        (True, False, False, BASESENTENCE.upper(), BASESENTENCE.lower(), 1.0),
        (False, False, False, BASESENTENCE.upper(), BASESENTENCE.lower(), 0.0),
        (False, True, False, "." + BASESENTENCE + ".", BASESENTENCE, 1.0),
        (False, False, False, "." + BASESENTENCE + ".", BASESENTENCE, 0.0),
        (False, False, True, BASESENTENCE, "The price of the shirt is 9.16", 1.0),
        (False, False, False, BASESENTENCE, "The price of the shirt is 9.16", 0.0),
    ]
)
def test_exact_matching(
    ignore_case, ignore_punctuation, ignore_numbers, prediction, reference, score) -> None:
    exact_match_string_evaluator = load_evaluator(
        evaluator=EvaluatorType.EXACT_MATCH,
        ignore_case=ignore_case, 
        ignore_punctuation=ignore_punctuation, 
        ignore_numbers=ignore_numbers
    )
    result = exact_match_string_evaluator.evaluate_strings(
        prediction=prediction, reference=reference
    )
    assert result["score"] == score, (f"prediction is <{prediction}>, reference is <{reference}>")

@pytest.mark.parametrize(
    "flags, prediction, reference, score", 
    [
        (0, BASESENTENCE, BASESENTENCE, 1.0),
        (0, BASESENTENCE.upper(), BASESENTENCE.lower(), 0.0),
        (re.I, BASESENTENCE.upper(), BASESENTENCE.lower(), 1.0),
        (0, BASESENTENCE+"I like it", BASESENTENCE, 1.0),
        (0, BASESENTENCE, BASESENTENCE+"I like it", 0.0),
    ]
)
def test_regex_matching(
    flags, prediction, reference, score) -> None:
    regex_match_string_evaluator = load_evaluator(
        evaluator=EvaluatorType.REGEX_MATCH,
        flags=flags
    )
    result = regex_match_string_evaluator.evaluate_strings(
        prediction=prediction, reference=reference
    )
    assert result["score"] == score, (f"prediction is <{prediction}>, reference is <{reference}>")