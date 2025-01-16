import os
from pathlib import Path

from loguru import logger

from prompt_scope.core.augmentor.demonstration_augmentation.ipc_aug import IPCGeneration
from prompt_scope.core.utils.utils import get_current_date
from prompt_scope.core.utils.utils import load_yaml
from prompt_scope.core.llms.dashscope_llm import DashscopeLLM, DashScopeLlmName

current_file_path = Path(__file__)
logger.add(f"{current_file_path.parent}/log/{current_file_path.stem}_{get_current_date()}.log", rotation="10 MB",
           level="INFO")
basic_config_path = os.path.join(os.path.dirname(__file__), 'ipc_aug_cn.yml')

config_params = load_yaml(basic_config_path)
generation_llm_name = DashScopeLlmName.QWEN_MAX
generation_llm = DashscopeLLM(model=generation_llm_name, temperature=0.1)
logger.info(config_params)
# Initializing the pipeline
pipeline = IPCGeneration(generation_llm=generation_llm, **config_params)
best_prompt = pipeline.run()