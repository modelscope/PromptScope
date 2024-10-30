import pytest
from pathlib import Path
from typing import List, Dict, Any, Set, Sequence, Literal
import pandas as pd
import numpy as np

from prompt_scope.core.utils.utils import load_yaml
from prompt_scope.core.utils.prompt_handler import PromptHandler
from prompt_scope.core.offline.instruction_optimization.prompt_agent.prompt_agent import PromptAgent
from prompt_scope.core.offline.instruction_optimization.prompt_agent.tasks import get_task
from prompt_scope.core.llms.dashscope_llm import DashscopeLLM
from prompt_scope.core.models.search_algo.loading import load_search_algo
from prompt_scope.core.models.world_model.loading import load_world_model


config_path = Path(__file__).parent.joinpath("prompt_agent_en.yml")
config_params = load_yaml(config_path)
base_llm = DashscopeLLM()
optim_llm = DashscopeLLM()
task = get_task(config_params["task_name"])(**config_params)
world_model = load_world_model(
    model=config_params["world_model_name"], 
    task=task,
    base_model=base_llm,
    optim_model=optim_llm,
    **config_params)
search_algo = load_search_algo(
    algo=config_params["search_algo_name"], 
    task=task,
    world_model=world_model,
    **config_params)
pagent = PromptAgent(task=task, search_algo=search_algo, **config_params)


def test_before_run():
    pagent._before_run()
    assert len(pagent.search_algo.world_model.nodes) == 1

def test_predict():
    pagent._before_run()
    path = pagent._predict()

def test_update_prompt():
    pagent._before_run()
    path = pagent._predict()
    x = pagent._update_prompt(path=path)
    
#     assert isinstance(path, Sequence)
#     assert isinstance(path[0], Sequence)
#     assert isinstance(raw_prompts_flattened, Sequence)
#     assert isinstance(raw_prompts_flattened_second_round, Sequence)
#     assert len(raw_answers) == len(raw_prompts_flattened) == 2
#     assert len(raw_answers_second_round) == len(raw_prompts_flattened_second_round)
#     if opro.extract_final_answer_by_prompting_again:
#         assert len(raw_answers) == len(raw_answers_second_round)
#         assert len(raw_prompts_flattened) == len(raw_prompts_flattened_second_round)

# def test_evaluate_and_analyze(): 
#     choices, accuracies = opro._evaluate_and_analyze(
#                 raw_answers=["160", "36000"], 
#                 true_answers=["160", "36000"],
#                 instruction="Please think step by step",
#                 prediction_treat_as_number=True,
#                 prediction_treat_as_bool=False,
#                 is_multiple_choice_all=False,
#                 )
    
#     assert isinstance(choices, Sequence)
#     assert isinstance(accuracies, Sequence)
#     assert len(choices) == len(accuracies) == 2
    
# def test_evaluate_single_instruction():
#     data_dict = load_data(dataset_name=opro.dataset_name, task_name=opro.task_name)
#     raw_data = data_dict['raw_data']
#     metadata = opro.evaluate_single_instruction(
#                                     instruction="Please think step by step",
#                                     raw_data=raw_data,
#                                     index_to_evaluate=[0,1],
#                                     true_answers=["160", "36000"],
#                                     prediction_treat_as_number=True,
#                                     prediction_treat_as_bool=False,
#                                     is_multiple_choice=False,
#                                     )
# def test_before_run():
#     opro._before_run()

# def test_update_prompt():
#     old_md5 = set()
#     new_md5, new_instructions, instructions_raw = opro._update_prompt(
#         i_step=0, meta_prompt="generate a prompt for writing", temperature=0.2,
#         old_instruction_md5_hashstrings_set=old_md5)
#     assert isinstance(new_instructions, Sequence)
#     assert len(new_md5) > 0
    