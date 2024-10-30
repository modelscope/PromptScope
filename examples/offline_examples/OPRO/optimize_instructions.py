import os
from pathlib import Path

from loguru import logger

from prompt_scope.core.offline.instruction_optimization.opro.opro import OPRO
from prompt_scope.core.utils.utils import get_current_date, load_yaml
from prompt_scope.core.llms.dashscope_llm import DashscopeLLM, DashScopeLlmName

current_file_path = Path(__file__)
logger.add(f"{current_file_path.parent}/log/{current_file_path.stem}_{get_current_date()}.log", rotation="10 MB",
           level="INFO")
basic_config_path = os.path.join(os.path.dirname(__file__), 'opro_en.yml')

optim_llm = DashscopeLLM(model=DashScopeLlmName.QWEN2_72B_INST)
config_params = load_yaml(basic_config_path)
logger.info(config_params)
# Initializing the pipeline
pipeline = OPRO(optim_llm=optim_llm, **config_params)
best_prompt = pipeline.run()
logger.info(best_prompt)
