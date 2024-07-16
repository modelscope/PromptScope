from meta_icl.core.offline.demonstration_augmentation.ipc import IPC_Generation
import argparse
import os
from pathlib import Path
import json

from meta_icl.core.utils.ipc_config import load_yaml
from meta_icl.core.enumeration.language_enum import LanguageEnum
from meta_icl.core.utils.logger import Logger

# General Training Parameters
# parser = argparse.ArgumentParser()

# parser.add_argument('--prompt',
#                     default='',
#                     required=False, type=str, help='Prompt to use as initial.')
# parser.add_argument('--task_description',
#                     default='',
#                     required=False, type=str, help='Describing the task')
# parser.add_argument('--output_dump', default='dump', required=False, type=str, help='Output to save checkpoints')
# parser.add_argument('--num_samples', default=10, type=int, help='Total Number of samples to generate')
# parser.add_argument('--batch_size', default=10, type=int, help='Number of samples in each batch')
# parser.add_argument('--language', default='cn', type=str, help='Language used by the prompt')
# parser.add_argument('--workers', default=5, type=int, help='workers for async call')

# opt = parser.parse_args()

logger = Logger.get_logger(__name__)
basic_config_path = 'ipc_aug_cn.yml'

config_params = load_yaml(basic_config_path)
logger.info(config_params)

# check language
# if opt.language.upper() != config_params.language.upper():
#     raise ValueError("Language inconsistency!")
if not hasattr(LanguageEnum, config_params.task_config.language.upper()):
    raise NotImplementedError("Only supports 'EN' and 'CN' for now!")

# if opt.task_description == '':
#     task_description = input("Describe the task: ")
# else:
#     task_description = opt.task_description

# if opt.prompt == '':
#     initial_prompt = input("Initial prompt: ")
# else:
#     initial_prompt = opt.prompt

# Initializing the pipeline
pipeline = IPC_Generation(config_params)
samples = pipeline.generate(config_params)

res = []
for sample in samples:
    num, question, answer = sample.split('\n')
    res.append({
        'ID': num,
        '问题': question,
        '答案': answer
    })
output_path = config_params.task_config.output_path
if output_path != '':
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    output_path = Path(output_path)
    with open(output_path / 'samples.json', 'w+') as f:
        for entry in res:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    



