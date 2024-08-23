import os
from pathlib import Path
import json
import pickle as pkl

from meta_icl.core.utils.utils import load_yaml
from meta_icl.core.enumeration.language_enum import LanguageEnum
from loguru import logger
from meta_icl import CONFIG_REGISTRY
from meta_icl.core.offline.instruction_optimization.ipc import IPC_Optimization
from meta_icl.core.utils.utils import get_current_date

current_file_path = Path(__file__)

logger.add(f"{current_file_path.parent}/log/{current_file_path.stem}_{get_current_date()}.log", rotation="10 MB", level="INFO")

rank_config_path = os.path.join(os.path.dirname(__file__), 'ipc_ranker.yml')
generate_config_path = os.path.join(os.path.dirname(__file__), 'ipc_optim_generate.yml')


rank_config_params = load_yaml(rank_config_path)
generate_config_params = load_yaml(generate_config_path)
logger.info(rank_config_params)
logger.info(generate_config_params)
CONFIG_REGISTRY.batch_register(rank_config_params)

if not hasattr(LanguageEnum, rank_config_params.task_config.language.upper()):
    raise NotImplementedError("Only supports 'EN' and 'CN' for now!")

kwargs, best_prompt = {}, None
if hasattr(generate_config_params.task_config, 'input_path'):
    input_path = Path(os.path.join(generate_config_params.task_config.input_path, 'ranking'))
    if (input_path / 'history.pkl').is_file():
        state = pkl.load(open(input_path / 'history.pkl', 'rb'))
        best_prompt = state['prompt']
        kwargs['ranking_prompt'] = state['prompt']

# Initializing the pipeline
pipeline = IPC_Optimization()

if not best_prompt:
    kwargs['mode'] = 'ranking'
    best_prompt = pipeline.run(**kwargs)

kwargs['mode'] = 'generation'
pipeline.samples = None

CONFIG_REGISTRY.batch_register(generate_config_params)
# CONFIG_REGISTRY.module_dict['eval_config'].instruction = best_prompt
pipeline = IPC_Optimization()
pipeline.init_config()
best_prompt = pipeline.run(**kwargs)
