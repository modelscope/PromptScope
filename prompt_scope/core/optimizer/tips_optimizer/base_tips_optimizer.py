import asyncio
import os
from pathlib import Path
from typing import List, Callable, Tuple, Dict
from abc import ABC
import asyncio
import nest_asyncio
nest_asyncio.apply()


from prompt_scope.core.prompt_gen.prompt_gen import BasePromptGen
from prompt_scope.core.llms.base import BaseLLM
from ..base_optimizer import BaseOptimizer

current_file_dir = Path(__file__).parent


class BaseTipsOptimizer(BaseOptimizer):
    def __init__(self,
                 infer_llm: BaseLLM,
                 optim_llm: BaseLLM,
                 init_system_prompt: str = None,
                 train_set: List[Dict] = None,
                 test_set: List[Dict] = None,
                 bad_case_analysis_prompt_dir: str = None,
                 details_save_dir: str = "./",
                 is_good_case_func: Callable[[str, str], Tuple] = None,
                 resume_generate: bool = True,
                 language: str = "cn",
                 epoch: int = 1,
                 train_bach_size: int = 5,
                 save_steps: int = 5,
                 max_workers_num: int = 10
                 ):
        super().__init__(
            infer_llm = infer_llm,
            optim_llm = optim_llm,
            init_system_prompt = init_system_prompt,
            train_set = train_set,
            test_set = test_set,
            details_save_dir = details_save_dir,
            is_good_case_func = is_good_case_func,
            language = language,
            epoch = epoch,
            train_bach_size= train_bach_size,
            save_steps = save_steps,
            max_workers_num = max_workers_num
        )

        self.bad_case_analysis_prompt_dir = bad_case_analysis_prompt_dir
        self.resume_generate = resume_generate

    def _init_optimizer_config(self):
        super()._init_optimizer_config()
        if self.bad_case_analysis_prompt_dir is None:
            self.bad_case_analysis_prompt_dir = \
                f"{current_file_dir}/prompt_lib/{self.language}/bad_case_analysis_prompt_{self.language}"

        self.bad_case_analysis_prompt = BasePromptGen.load(promptgen_load_dir=self.bad_case_analysis_prompt_dir)

    def _extract_tips(self, case_analysis: str):
        if "<tips>" in case_analysis and "</tips>" in case_analysis:
            tips = case_analysis.split("<tips>")[-1].split("</tips>")[0].strip()
            return tips
        else:
            return None


