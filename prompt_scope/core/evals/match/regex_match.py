import re
from typing import Any, List

from prompt_scope.core.evals.schema import StringEvaluator


class RegexMatchStringEvaluator(StringEvaluator):
    """Compute a regex match between the prediction and the reference"""

    def __init__(self, *, flags: int = 0, **kwargs: Any):
        super().__init__()
        self.flags = flags

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
        """This evaluator requires a reference."""
        return True

    @property
    def input_keys(self) -> List[str]:
        """
        Get the input keys.

        Returns:
            List[str]: The input keys.
        """
        return ["reference", "prediction"]

    @property
    def evaluation_name(self) -> str:
        """
        Get the evaluation name.

        Returns:
            str: The evaluation name.
        """
        return "regex_match"

    def _evaluate_strings(
        self,
        *,
        prediction: str,
        reference: str,
        input: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """
        Evaluate the regex match between the prediction and the reference.

        Args:
            prediction (str): The prediction string.
            reference (Optional[str], optional): The reference regex pattern.

        Returns:
            dict: The evaluation results containing the score.
        """
        match = re.match(reference, prediction, flags=self.flags)
        return {"score": int(bool(match))}
