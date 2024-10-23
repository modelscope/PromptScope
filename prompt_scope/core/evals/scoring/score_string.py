from __future__ import annotations

from typing import Any, Optional
from pydantic import Field
from loguru import Logger

from prompt_scope.core.models.base_model import BaseModel
from prompt_scope.core.evals.schema import StringEvaluator
from prompt_scope.core.evals.scoring.prompts import SCORING_PROMPT
from prompt_scope.core.models.generation_model import GenerationModel

from . import ScoreSchema


logger = Logger.get_logger()
        
class ScoreStringEvaluator(StringEvaluator):
    """score on a scale of 1-10 the output of a LLM."""

    @property
    def requires_input(self) -> bool:
        """This evaluator requires an input."""
        return True

    @property
    def requires_llm(self) -> bool:
        """This evaluator requires a llm."""
        return True

    @property
    def requires_reference(self) -> bool:
        """This evaluator does not require a reference."""
        return False
    
    @property
    def evaluation_name(self) -> str:
        """Get the name of the evaluation."""
        return "score_string"

    @property
    def _skip_reference_warning(self) -> str:
        """Return the warning to show when reference is ignored.

        Returns:
            str: The warning to show when reference is ignored.

        """
        return (
            f"Ignoring reference in {self.__class__.__name__}, as it is not expected."
            "\nTo use a reference, use the LabeledScoreStringEvalChain instead."
            " (EvaluatorType.LABELED_SCORE_STRING) instead."
        )

    def _evaluate_strings(
        self,
        prediction: str,
        llm: BaseModel = Field(default_factory=GenerationModel, description="llm to use."),
        input: Optional[str] = None,
        reference: Optional[str] = None,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Score the output string.

        Args:
            prediction (str): The output string from the first model.
            input (str, optional): The input or task string.
            reference (str, optional): The reference is not needed in ScoreStringEvaluator.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - comment: Comments for the answer and the score.
                - score: A score between 1 and 10.
        """
        prompt_ = prompt or SCORING_PROMPT.format(input=input,prediction=prediction)
        
        return self.llm.structured_output(messages=prompt_, schema=ScoreSchema, example_instruction=False)
    