import pytest
from unittest.mock import MagicMock, create_autospec
from prompt_scope.core.offline.instruction_optimization.ipc.ipc import IPCOptimization, PredictSchema, SampleSchema
from prompt_scope.core.evals.loading import load_evaluator
from prompt_scope.core.utils.utils import load_yaml
import os
from pathlib import Path
from typing import List, Dict, Any

# 测试数据
samples = ["这部电影揭示了主角的全部背景故事，对于接下来的发展有重大影响。", "观众对这周上映的新片有极高的期待值。"]
reference = [PredictSchema(prediction="是", sample="这部电影揭示了主角的全部背景故事，对于接下来的发展有重大影响。"), 
             PredictSchema(prediction="否", sample="观众对这周上映的新片有极高的期待值。")]
prediction = [PredictSchema(prediction="是", sample="这部电影揭示了主角的全部背景故事，对于接下来的发展有重大影响。"), 
              PredictSchema(prediction="是", sample="观众对这周上映的新片有极高的期待值。")]
label_schema = ["是", "否"]

# 创建一个 IPCOptimization 实例
config_path = Path(__file__).parent.joinpath("ipc_classify_cn.yml")
config_params = load_yaml(config_path)
ipc_optimization = IPCOptimization(**config_params)

def test_generate():
    prompt_input = {
        'task_description': ipc_optimization.task_description, 
        'instruction': ipc_optimization.instruction,
        'batch_size': ipc_optimization.batch_size
        }
    generate_prompt = getattr(ipc_optimization.prompt_handler, 'adv_sample_classification') \
        .format_map(prompt_input)
    samples = ipc_optimization._generate(
        prompt=generate_prompt
    )
    assert isinstance(samples, List)
    assert isinstance(samples[0], str)
    assert len(samples) == ipc_optimization.samples_per_step

def test_predict():
    prediction = ipc_optimization._predict(
        samples=samples,
        llm=ipc_optimization.predictor_llm
    )
    assert isinstance(prediction, List)
    assert isinstance(prediction[0], PredictSchema)
    assert len(prediction) == 2

def test_annotation():
    annotation = ipc_optimization._predict(
        samples=samples,
        llm=ipc_optimization.annotate_llm
    )
    assert isinstance(annotation, List)
    assert isinstance(annotation[0], PredictSchema)
    assert len(annotation) == 2

def test_evaluate_and_analyze():
    history = ipc_optimization._evaluate_and_analyze(
        reference=reference,
        prediction=prediction,
        evaluator=load_evaluator('exact_match'),
        label_schema=label_schema
    )
    assert isinstance(history, List)
    assert isinstance(history[0], Dict)
    assert len(history) == 1
    
def test_step_generate():
    extra_samples_text = '##\n'
    for sample in samples:
        extra_samples_text += f"Sample:\n {sample}\n#\n"
    prompt_input = {
        'history': 'No previous errors information', 
        'instruction': ipc_optimization.instruction,
        'batch_size': ipc_optimization.batch_size,
        'task_description': ipc_optimization.task_description, 
        'extra_samples': extra_samples_text
        }
    generate_prompt = ipc_optimization.prompt_handler.step_adv_sample_classification.format_map(prompt_input)
    new_samples = ipc_optimization._generate(
        prompt=generate_prompt
    )
    assert isinstance(new_samples, List)
    assert isinstance(new_samples[0], str)
    assert len(new_samples) == ipc_optimization.samples_per_step

# 运行测试
if __name__ == "__main__":
    pytest.main()