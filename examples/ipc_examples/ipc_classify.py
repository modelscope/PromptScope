import os
from pathlib import Path

from loguru import logger

from prompt_scope.core.optimizer.research_optimizers.ipc_optimizer.ipc import IPCOptimization
from prompt_scope.core.utils.utils import get_current_date
from prompt_scope.core.utils.utils import load_yaml

current_file_path = Path(__file__)
logger.add(f"{current_file_path.parent}/log/{current_file_path.stem}_{get_current_date()}.log", rotation="10 MB",
           level="INFO")
basic_config_path = os.path.join(os.path.dirname(__file__), 'ipc_classify_cn.yml')

config_params = load_yaml(basic_config_path)
logger.info(config_params)
# Initializing the pipeline
pipeline = IPCOptimization(**config_params)
best_prompt = pipeline.run()