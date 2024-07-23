from meta_icl.core.offline.instruction_optimization.ipc_classifier import IPC_Optimization
import os
from pathlib import Path
import json
import pickle as pkl

from meta_icl.core.utils.ipc_config import load_yaml
from meta_icl.core.enumeration.language_enum import LanguageEnum
from meta_icl.core.utils.logger import Logger
from meta_icl import CONFIG_REGISTRY

logger = Logger.get_logger(__name__)
rank_config_path = 'ipc_ranker.yml'
generate_config_path = 'ipc_optim_generate.yml'

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
    if (input_path / 'samples.json').is_file():
        with open(input_path / 'samples.json', 'r') as f:
            data = [json.loads(line) for line in f]
        kwargs['data'] = data
    if (input_path / 'history.pkl').is_file():
        state = pkl.load(open(input_path / 'history.pkl', 'rb'))
        best_prompt = state['prompt']
        kwargs['ranking_prompt'] = state['prompt']

# Initializing the pipeline
pipeline = IPC_Optimization()

if not best_prompt:
    kwargs['mode'] = 'ranking'
    best_prompt = pipeline.run_pipeline(**kwargs)

kwargs['mode'] = 'generation'
pipeline.samples = None

# config_params = load_yaml(generate_config_path)
# logger.info(config_params)
CONFIG_REGISTRY.batch_register(generate_config_params)
# CONFIG_REGISTRY.module_dict['eval_config'].instruction = best_prompt
pipeline = IPC_Optimization()
pipeline.init_config()
best_prompt = pipeline.run_pipeline(**kwargs)
