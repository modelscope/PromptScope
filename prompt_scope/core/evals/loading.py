"""Loading datasets and evaluators."""

from typing import Any, Dict, Type

from prompt_scope.core.evals.match.exact_match import ExactMatchStringEvaluator
from prompt_scope.core.evals.match.regex_match import RegexMatchStringEvaluator
from prompt_scope.core.evals.parsing.json_validity import JsonValidityEvaluator
from prompt_scope.core.evals.parsing.json_schema import JsonSchemaEvaluator
from prompt_scope.core.evals.scoring.score_string import ScoreStringEvaluator
from prompt_scope.core.evals.scoring.labeled_score_string import LabeledScoreStringEvaluator
from prompt_scope.core.evals.schema import EvaluatorType, StringEvaluator


_EVALUATOR_MAP: Dict[
    EvaluatorType, Type[StringEvaluator]] = {
    EvaluatorType.SCORE_STRING: ScoreStringEvaluator,
    EvaluatorType.LABELED_SCORE_STRING: LabeledScoreStringEvaluator,
    EvaluatorType.JSON_VALIDITY: JsonValidityEvaluator,
    EvaluatorType.JSON_SCHEMA_VALIDATION: JsonSchemaEvaluator,
    EvaluatorType.REGEX_MATCH: RegexMatchStringEvaluator,
    EvaluatorType.EXACT_MATCH: ExactMatchStringEvaluator,
}


def load_evaluator(
    evaluator: EvaluatorType,
    **kwargs: Any,
) -> StringEvaluator:
    """Load the requested evaluator specified by a string.

    Parameters
    ----------
    evaluator : EvaluatorType
        The type of evaluator to load.
    llm : BaseLanguageModel, optional
        The language model to use for evaluation, by default None
    **kwargs : Any
        Additional keyword arguments to pass to the evaluator.

    Returns
    -------
    StringEvaluator
        The loaded evaluator.
    """
    if evaluator not in _EVALUATOR_MAP:
        raise ValueError(
            f"Unknown evaluator type: {evaluator}"
            f"\nValid types are: {list(_EVALUATOR_MAP.keys())}"
        )
    evaluator_cls = _EVALUATOR_MAP[evaluator]
    return evaluator_cls(**kwargs)