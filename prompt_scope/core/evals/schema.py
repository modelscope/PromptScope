"""Interfaces to be implemented by general evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Union
from warnings import warn
import asyncio
from pydantic import Field


from loguru import logger
from prompt_scope.core.models.base_model import BaseModel
from prompt_scope.core.models.generation_model import GenerationModel


class EvaluatorType(str, Enum):
    """The types of the evaluators."""

    SCORE_STRING = "score_string"
    """The scored string evaluator, which gives a score between 1 and 10 
    to a prediction."""
    LABELED_SCORE_STRING = "labeled_score_string"
    """The labeled scored string evaluator, which gives a score between 1 and 10
    to a prediction based on a ground truth reference label."""
    EXACT_MATCH = "exact_match"
    """Compare predictions to a reference answer using exact matching."""
    REGEX_MATCH = "regex_match"
    """Compare predictions to a reference answer using regular expressions."""
    JSON_VALIDITY = "json_validity"
    """Check if a prediction is valid JSON."""
    JSON_SCHEMA_VALIDATION = "json_schema_validation"
    """Check if a prediction is valid JSON according to a JSON schema."""


class _EvalArgsMixin:
    """Mixin for checking evaluation arguments."""

    @property
    def requires_reference(self) -> bool:
        """Whether this evaluator requires a reference label."""
        return False

    @property
    def requires_input(self) -> bool:
        """Whether this evaluator requires an input string."""
        return False

    @property
    def _skip_input_warning(self) -> str:
        """Warning to show when input is ignored."""
        return f"Ignoring input in {self.__class__.__name__}, as it is not expected."

    @property
    def _skip_reference_warning(self) -> str:
        """Warning to show when reference is ignored."""
        return (
            f"Ignoring reference in {self.__class__.__name__}, as it is not expected."
        )

    def _check_evaluation_args(
        self,
        reference: Optional[str] = None,
        input: Optional[str] = None,
    ) -> None:
        """Check if the evaluation arguments are valid.

        Args:
            reference (Optional[str], optional): The reference label.
            input (Optional[str], optional): The input string.
        Raises:
            ValueError: If the evaluator requires an input string but none is provided,
                or if the evaluator requires a reference label but none is provided.
        """
        if self.requires_input and input is None:
            raise ValueError(f"{self.__class__.__name__} requires an input string.")
        elif input is not None and not self.requires_input:
            warn(self._skip_input_warning)
        if self.requires_reference and reference is None:
            raise ValueError(f"{self.__class__.__name__} requires a reference string.")
        elif reference is not None and not self.requires_reference:
            warn(self._skip_reference_warning)


class StringEvaluator(_EvalArgsMixin, ABC):
    """Grade, tag, or otherwise evaluate predictions relative to their inputs
    and/or reference labels."""

    @property
    def evaluation_name(self) -> str:
        """The name of the evaluation."""
        return self.__class__.__name__

    @property
    def requires_input(self) -> bool:
        """Whether this evaluator requires an input."""
        return False
    
    @property
    def requires_llm(self) -> bool:
        """Whether this evaluator requires a llm."""
        return False
    
    @property
    def requires_reference(self) -> bool:
        """Whether this evaluator requires a reference label."""
        return False

    def _prepare_llm(
        self, 
        llm: BaseModel = Field(default_factory=GenerationModel, description="llm to use.")
    ):
        self.llm = llm()

    @abstractmethod
    def _evaluate_strings(
        self,
        *,
        prediction: Union[str, Any],
        reference: Optional[Union[str, Any]] = None,
        input: Optional[Union[str, Any]] = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate Chain or LLM output, based on optional input and label.

        Args:
            prediction (str): The LLM or chain prediction to evaluate.
            reference (Optional[str], optional): The reference label to evaluate against.
            input (Optional[str], optional): The input to consider during evaluation.
            kwargs: Additional keyword arguments, including callbacks, tags, etc.
        Returns:
            dict: The evaluation results containing the score or value.
                It is recommended that the dictionary contain the following keys:
                     - score: the score of the evaluation, if applicable.
                     - value: the string value of the evaluation, if applicable.
                     - reasoning: the reasoning for the evaluation, if applicable.
        """

    async def _aevaluate_strings(
        self,
        *,
        prediction: Union[str, Any],
        reference: Optional[Union[str, Any]] = None,
        input: Optional[Union[str, Any]] = None,
        **kwargs: Any,
    ) -> dict:
        """Asynchronously evaluate Chain or LLM output, based on optional input and label.

        Args:
            prediction (str): The LLM or chain prediction to evaluate.
            reference (Optional[str], optional): The reference label to evaluate against.
            input (Optional[str], optional): The input to consider during evaluation.
            kwargs: Additional keyword arguments, including callbacks, tags, etc.
        Returns:
            dict: The evaluation results containing the score or value.
                It is recommended that the dictionary contain the following keys:
                     - score: the score of the evaluation, if applicable.
                     - value: the string value of the evaluation, if applicable.
                     - reasoning: the reasoning for the evaluation, if applicable.
        """
        result = await asyncio.to_thread(
            self.evaluate_strings, prediction=prediction, reference=reference, input=input, **kwargs
        )
        return result

    def evaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate Chain or LLM output, based on optional input and label.

        Args:
            prediction (str): The LLM or chain prediction to evaluate.
            reference (Optional[str], optional): The reference label to evaluate against.
            input (Optional[str], optional): The input to consider during evaluation.
            kwargs: Additional keyword arguments, including callbacks, tags, etc.
        Returns:
            dict: The evaluation results containing the score or value.
        """
        self._check_evaluation_args(reference=reference, input=input)
        if self.requires_llm and not hasattr(self, 'llm'):
            self._prepare_llm(llm=kwargs.get('llm', GenerationModel))
        return self._evaluate_strings(
            prediction=prediction, reference=reference, input=input, **kwargs
        )
    

    async def aevaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Asynchronously evaluate Chain or LLM output, based on optional input and label.

        Args:
            prediction (str): The LLM or chain prediction to evaluate.
            reference (Optional[str], optional): The reference label to evaluate against.
            input (Optional[str], optional): The input to consider during evaluation.
            kwargs: Additional keyword arguments, including callbacks, tags, etc.
        Returns:
            dict: The evaluation results containing the score or value.
        """
        self._check_evaluation_args(reference=reference, input=input)
        return await self._aevaluate_strings(
            prediction=prediction, reference=reference, input=input, **kwargs
        )