from meta_icl.core.offline.instruction_optimization.ipc_classifier import IPC_Optimization
import argparse
import os
from pathlib import Path
import json

from meta_icl.core.utils.ipc_config import load_yaml
from meta_icl.core.enumeration.language_enum import LanguageEnum
from meta_icl.core.utils.logger import Logger
from meta_icl import CONFIG_REGISTRY

logger = Logger.get_logger(__name__)
basic_config_path = 'ipc_optim_classify.yml'

config_params = load_yaml(basic_config_path)
logger.info(config_params)

CONFIG_REGISTRY.batch_register(config_params)

if not hasattr(LanguageEnum, config_params.task_config.language.upper()):
    raise NotImplementedError("Only supports 'EN' and 'CN' for now!")

kwargs = {}
if hasattr(config_params.task_config, 'input_path'):
    input_path = Path(config_params.task_config.input_path)
    if os.path.exists(input_path / 'samples.json'):
        with open(input_path / 'samples.json', 'r') as f:
            data = [json.loads(line) for line in f]
        kwargs['data'] = data

# Initializing the pipeline
pipeline = IPC_Optimization()
best_prompt = pipeline.run_pipeline(**kwargs)

# res = []
# for sample in samples:
#     num, question, answer = sample.split('\n')
#     res.append({
#         'ID': num,
#         '问题': question,
#         '答案': answer
#     })
# output_path = config_params.task_config.output_path
# if output_path != '':
#     if not os.path.isdir(output_path):
#         os.makedirs(output_path)
#     output_path = Path(output_path)
#     with open(output_path / 'samples.json', 'w+') as f:
#         for entry in res:
#             f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    



