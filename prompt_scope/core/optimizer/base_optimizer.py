import asyncio
import os
from pathlib import Path
from typing import List, Callable, Tuple, Dict
from abc import ABC
import asyncio
import nest_asyncio
nest_asyncio.apply()


from prompt_scope.core.llms.base import BaseLLM
from prompt_scope.core.schemas.example import LLMCallRecord

current_file_dir = Path(__file__).parent


class BaseOptimizer(ABC):
    def __init__(self,
                 infer_llm: BaseLLM,
                 optim_llm: BaseLLM,
                 init_system_prompt: str = None,
                 train_set: List[Dict] = None,
                 test_set: List[Dict] = None,
                 details_save_dir: str = "./",
                 is_good_case_func: Callable[[str, str], Tuple] = None,
                 language: str = "cn",
                 epoch: int = 1,
                 train_bach_size: int = 5,
                 save_steps: int = 5,
                 max_workers_num: int = 10
                 ):
        self.infer_llm = infer_llm
        self.optim_llm = optim_llm
        self.init_system_prompt = init_system_prompt

        self.train_set = [LLMCallRecord(**d) for d in train_set]
        self.test_set = [LLMCallRecord(**d) for d in test_set]
        self.details_save_dir = details_save_dir
        self.optimizer_config = None
        self.is_good_case_func = is_good_case_func

        self.bad_case_analysis_prompt = None
        self.tips_postprocess_prompt = None
        self.language = language
        self.epoch = epoch
        self.train_bach_size = train_bach_size
        self.save_steps = save_steps
        self.semaphore = asyncio.Semaphore(max_workers_num)

    def _init_optimizer_config(self):
        if not os.path.exists(self.details_save_dir):
            os.makedirs(self.details_save_dir)

    def _before_run(self):
        pass

    def run(self,):
        pass

    def _after_run(self, history_scores, history_prompts):
        pass



