import os
from pathlib import Path

from loguru import logger

from prompt_scope.core.optimizer.research_optimizers.opro_optimizer.opro import OPRO
from prompt_scope.core.utils.utils import get_current_date, load_yaml
from prompt_scope.core.llms.dashscope_llm import DashscopeLLM, DashScopeLlmName

current_file_path = Path(__file__)
logger.add(f"{current_file_path.parent}/log/{current_file_path.stem}_{get_current_date()}.log", rotation="10 MB",
           level="INFO")
basic_config_path = os.path.join(os.path.dirname(__file__), 'opro_cn.yml')

scorer_llm_name = DashScopeLlmName.QWEN_PLUS
optim_llm_name = DashScopeLlmName.QWEN_MAX
optim_llm = DashscopeLLM(model=optim_llm_name, temperature=0.1)
scorer_llm = DashscopeLLM(model=scorer_llm_name, temperature=0.0)
config_params = load_yaml(basic_config_path)
logger.info(config_params)
# Initializing the pipeline
pipeline = OPRO(scorer_llm_name=scorer_llm_name, 
                optim_llm_name=optim_llm_name, 
                scorer_llm=scorer_llm, 
                optim_llm=optim_llm, 
                **config_params)

best_prompt = pipeline.run()
logger.info(best_prompt)