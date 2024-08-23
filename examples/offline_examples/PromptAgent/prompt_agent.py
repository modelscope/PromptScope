import os
from pathlib import Path

from meta_icl.core.offline.instruction_optimization.prompt_agent import PromptAgent
from meta_icl.core.utils.utils import load_yaml
from loguru import logger

from meta_icl import CONFIG_REGISTRY
from meta_icl.core.utils.utils import get_current_date

current_file_path = Path(__file__)

logger.add(f"{current_file_path.parent}/log/{current_file_path.stem}_{get_current_date()}.log", rotation="10 MB", level="INFO")

def config():
    config_dir = os.path.join(os.path.dirname(__file__), "prompt_agent.yml")
    args = load_yaml(config_dir)
    return args

def main():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    agent = PromptAgent(dataset_path=os.path.join(cur_path, CONFIG_REGISTRY.module_dict['task_config'].data_dir))
    agent.run()

if __name__ == '__main__':
    args = config()
    logger.info(args)
    CONFIG_REGISTRY.batch_register(args)
    print(CONFIG_REGISTRY.module_dict)
    main()