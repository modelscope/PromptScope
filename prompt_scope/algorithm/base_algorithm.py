from abc import ABC, abstractmethod
from typing import List, Literal, Any, Dict
from pydantic import BaseModel, Field

from prompt_scope.core.utils.prompt_handler import PromptHandler
from prompt_scope.core.llms.base import BaseLLM
from prompt_scope.core.llms.dashscope_llm import DashscopeLLM
from prompt_scope.core.evals.loading import load_evaluator

class PromptOptimizationWithFeedback(BaseModel, ABC):
    """
    Base Abstract Class for Prompt Optimization with Feedback
    """
    # ============= LLM Configuration =============
    generation_llm: BaseLLM = Field(default=DashscopeLLM(max_retries=3))
    predictor_llm: BaseLLM = Field(default=DashscopeLLM(max_retries=1))
    analyzer_llm: BaseLLM = Field(default=DashscopeLLM(max_retries=1))

    # ============= Basic Configuration =============
    task_type: Literal["classification", "generation"] = Field(...)
    language: Literal["cn", "en"] = Field(default="cn")
    instruction: str = Field(default="", description="Initial instruction")
    num_steps: int = Field(default=5)
    patient: int = Field(default=0)  # Patience counter for optimization steps
    cur_step: int = Field(default=0)  # Tracks the current step in the iterative process
    prompt_path: str = Field(default="")
    history: List = []

    # ============= Experiment Configuration =============
    eval_type: Literal["score_string", "exact_match"] = Field(..., description="Evaluator Type for the LLM answer")
    result_dict: Dict[str, Any]= Field(default={})


    @property
    def evaluator(self):
        return load_evaluator(self.eval_type)
        
    @property
    def prompt_handler(self):
        """
        Returns:
            PromptHandler: An instance of PromptHandler initialized with specific file path and keyword arguments.
        """
        return PromptHandler(self.prompt_path, language=self.language)

    @abstractmethod
    def _before_run(self):
        pass
    
    @abstractmethod
    def _after_run(self):
        pass
    
    def run(self):
        self._before_run()
        for _ in range(self.num_steps):
            if self._step():
                break
        self._after_run()

    @abstractmethod
    def _step(self):
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
