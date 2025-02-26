import pytest
from pathlib import Path
from typing import List, Dict, Any, Set, Sequence, Literal
import pandas as pd
import numpy as np

from prompt_scope.core.utils.utils import load_yaml
from prompt_scope.core.optimizer.research_optimizers.opro_optimizer.utils import load_data
from prompt_scope.core.optimizer.research_optimizers.opro_optimizer.opro import OPRO

# 测试数据
# samples = ["这部电影揭示了主角的全部背景故事，对于接下来的发展有重大影响。", "观众对这周上映的新片有极高的期待值。"]
# reference = [PredictSchema(prediction="是", sample="这部电影揭示了主角的全部背景故事，对于接下来的发展有重大影响。"), 
#              PredictSchema(prediction="否", sample="观众对这周上映的新片有极高的期待值。")]
# prediction = [PredictSchema(prediction="是", sample="这部电影揭示了主角的全部背景故事，对于接下来的发展有重大影响。"), 
#               PredictSchema(prediction="是", sample="观众对这周上映的新片有极高的期待值。")]
# label_schema = ["是", "否"]

# 创建一个 IPCOptimization 实例
config_path = Path(__file__).parent.joinpath("opro_en.yml")
config_params = load_yaml(config_path)
opro = OPRO(**config_params)

def test_load_data():
    data_kwargs = load_data(dataset_name=opro.dataset_name, task_name=opro.task_name)
    assert isinstance(data_kwargs["raw_data"], pd.DataFrame | Sequence)
    assert isinstance(data_kwargs["prediction_treat_as_number"], bool | Literal["adaptive"])
    assert isinstance(data_kwargs["prediction_treat_as_bool"], bool | Literal["adaptive"])
    assert isinstance(data_kwargs["multiple_choice_tasks"], Set)

def test_predict():
    data_dict = load_data(dataset_name=opro.dataset_name, task_name=opro.task_name)
    raw_data = data_dict['raw_data']
    prediction_metadata = opro._predict(
        eval_index_all=[0,1], 
        raw_data=raw_data, 
        instruction="Please think step by step")
    
    raw_answers = prediction_metadata["raw_answers"]
    raw_answers_second_round = prediction_metadata.get("raw_answers_second_round", [])
    raw_prompts_flattened = prediction_metadata["raw_prompts_flattened"]
    raw_prompts_flattened_second_round = prediction_metadata.get("raw_prompts_flattened_second_round", [])


    assert isinstance(raw_answers, Sequence)
    assert isinstance(raw_answers_second_round, Sequence)
    assert isinstance(raw_prompts_flattened, Sequence)
    assert isinstance(raw_prompts_flattened_second_round, Sequence)
    assert len(raw_answers) == len(raw_prompts_flattened) == 2
    assert len(raw_answers_second_round) == len(raw_prompts_flattened_second_round)
    if opro.extract_final_answer_by_prompting_again:
        assert len(raw_answers) == len(raw_answers_second_round)
        assert len(raw_prompts_flattened) == len(raw_prompts_flattened_second_round)

def test_evaluate_and_analyze(): 
    choices, accuracies = opro._evaluate_and_analyze(
                raw_answers=["160", "36000"], 
                true_answers=["160", "36000"],
                instruction="Please think step by step",
                prediction_treat_as_number=True,
                prediction_treat_as_bool=False,
                is_multiple_choice_all=False,
                )
    
    assert isinstance(choices, Sequence)
    assert isinstance(accuracies, Sequence)
    assert len(choices) == len(accuracies) == 2
    
def test_evaluate_single_instruction():
    data_dict = load_data(dataset_name=opro.dataset_name, task_name=opro.task_name)
    raw_data = data_dict['raw_data']
    metadata = opro.evaluate_single_instruction(
                                    instruction="Please think step by step",
                                    raw_data=raw_data,
                                    index_to_evaluate=[0,1],
                                    true_answers=["160", "36000"],
                                    prediction_treat_as_number=True,
                                    prediction_treat_as_bool=False,
                                    is_multiple_choice=False,
                                    )
def test_before_run():
    opro._before_run()

def test_update_prompt():
    old_md5 = set()
    new_md5, new_instructions, instructions_raw = opro._update_prompt(
        i_step=0, meta_prompt="generate a prompt for writing", temperature=0.2,
        old_instruction_md5_hashstrings_set=old_md5)
    assert isinstance(new_instructions, Sequence)
    assert len(new_md5) > 0
    

# 运行测试
if __name__ == "__main__":
    pytest.main()