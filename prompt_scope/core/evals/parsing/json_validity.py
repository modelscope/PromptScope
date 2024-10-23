"""Evaluators for parsing strings."""

import json
from typing import Any, Optional

from prompt_scope.core.evals.parsing.utils import parse_json_markdown

from prompt_scope.core.evals.schema import StringEvaluator


class JsonValidityEvaluator(StringEvaluator):
    """Evaluate whether the prediction is valid JSON.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

    @property
    def requires_input(self) -> bool:
        """This evaluator does not require input."""
        return False
    
    @property
    def requires_llm(self) -> bool:
        """This evaluator does not require llm."""
        return False
    
    @property
    def requires_reference(self) -> bool:
        """This evaluator does not require reference."""
        return False

    @property
    def evaluation_name(self) -> str:
        return "json_validity"

    def _evaluate_strings(
        self,
        prediction: str,
        input: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate the prediction string.

        Args:
            prediction (str): The prediction string to evaluate.
            input (str, optional): Not used in this evaluator. Defaults to None.
            reference (str, optional): Not used in this evaluator. Defaults to None.

        Returns:
            dict: A dictionary containing the evaluation score. The score is 1 if
            the prediction is valid JSON, and 0 otherwise.
                If the prediction is not valid JSON, the dictionary also contains
                a "reasoning" field with the error message.

        """
        try:
            parse_json_markdown(prediction, parser=json.loads)
            return {"score": 1}
        except Exception as e:
            return {"score": 0, "reasoning": str(e)}