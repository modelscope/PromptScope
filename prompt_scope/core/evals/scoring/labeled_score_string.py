from __future__ import annotations

from typing import Any, Optional

from loguru import logger
from prompt_scope.core.evals.scoring.prompts import SCORING_PROMPT_WITH_REFERENCE
from prompt_scope.core.evals.scoring.score_string import ScoreStringEvaluator

from . import ScoreSchema


class LabeledScoreStringEvaluator(ScoreStringEvaluator):
    """scoring the output of a model on a scale of 1-10 according to the reference."""
    @property
    def requires_reference(self) -> bool:
        """This evaluator requires a reference."""
        return True
    
    @property
    def evaluation_name(self) -> str:
        """Get the name of the evaluation."""
        return "labeled_score_string"

    def _evaluate_strings(
        self,
        prediction: str,
        input: Optional[str] = None,
        reference: Optional[str] = None,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Score the output string.

        Args:
            prediction (str): The output string from the first model.
            input (str, optional): The input or task string.
            reference (str, optional): The reference string, if any.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - comment: Comments for the answer and the score.
                - score: A score between 1 and 10.
        """
        prompt_ = prompt or SCORING_PROMPT_WITH_REFERENCE.format(
            reference=reference,input=input,prediction=prediction)
        
        return self.llm.structured_output(messages=prompt_, schema=ScoreSchema, example_instruction=False)