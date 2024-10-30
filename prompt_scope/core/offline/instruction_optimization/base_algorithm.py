from abc import ABC, abstractmethod
from typing import List, Literal, Any, Dict
from pydantic import BaseModel, Field, PrivateAttr
import os

from prompt_scope.core.utils.prompt_handler import PromptHandler
from prompt_scope.core.llms.base import BaseLLM
from prompt_scope.core.evals.loading import load_evaluator
from prompt_scope.core.evals.schema import StringEvaluator

class PromptOptimizationWithFeedback(BaseModel, ABC):
    """
    Base Abstract Class for Prompt Optimization with Feedback
    """
    # ============= Basic Configuration =============
    language: Literal["cn", "en"] = Field(default="cn")
    instruction: str = Field(default="", description="Initial instruction")
    num_steps: int = Field(default=5)
    patient: int = Field(default=0)  # Patience counter for optimization steps

    # ============= Experiment Configuration =============
    eval_type: Literal["score_string", "exact_match", "regex_match", "custom"] = Field(..., description="Evaluator Type for the LLM answer")
    result_dict: Dict[str, Any]= Field(default={})
    _prompt_handler: PromptHandler | None = PrivateAttr(default=None)
    _evaluator: StringEvaluator | None = PrivateAttr(default=None)


    @property
    def evaluator(self) -> StringEvaluator:
        if self.eval_type == "custom":
            raise ValueError("evaluator is not supported when eval_type is custom")
        if not self._evaluator:
            self._evaluator = load_evaluator(self.eval_type)
        return self._evaluator
        
    @property
    def prompt_handler(self) -> PromptHandler:
        """
        Returns:
            PromptHandler: An instance of PromptHandler initialized with specific file path and keyword arguments.
        """
        if not self._prompt_handler:
            self._prompt_handler = PromptHandler(self.prompt_path, language=self.language)
        return self._prompt_handler

    @abstractmethod
    def _before_run(self):
        pass
    
    @abstractmethod
    def _after_run(self):
        pass
    
    def run(self) -> Dict[str, Any]:
        self._before_run()
        for i_step in range(self.num_steps):
            if self._step(i_step=i_step):
                break
        return self._after_run()

    @abstractmethod
    def _step(self) -> bool:
        pass

    @abstractmethod
    def _predict(self, *, samples: List[str], llm: BaseLLM):
        pass

    @abstractmethod
    def _evaluate_and_analyze(self, 
                              *, 
                              input: List[Any]=[], 
                              reference: List[Any]=[], 
                              prediction: List[Any],
                              evaluator: Any):
        pass

    @abstractmethod
    def _update_prompt(self):
        pass

    @abstractmethod
    def extract_best_prompt(self):
        pass


class DemonstrationAugmentation(ABC):
    """
    Base Abstract Class for Prompt Optimization with Feedback
    """

    def __init__(self):
        pass

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def init_config(self):
        pass

    @abstractmethod
    def init_prompt(self):
        pass

    @abstractmethod
    def run(self):
        pass
