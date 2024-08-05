import os
from pathlib import Path
import json

from meta_icl.core.utils.utils import load_yaml
from meta_icl.core.enumeration.language_enum import LanguageEnum
from meta_icl.core.utils.logger import Logger
from meta_icl import CONFIG_REGISTRY, PROMPT_REGISTRY
from meta_icl.core.offline.demonstration_augmentation.ipc_aug import IPC_Generation

logger = Logger.get_logger(__name__)
basic_config_path = 'ipc_aug_cn.yml'

config_params = load_yaml(basic_config_path)
logger.info(config_params)

CONFIG_REGISTRY.batch_register(config_params)

if not hasattr(LanguageEnum, config_params.task_config.language.upper()):
    raise NotImplementedError("Only supports 'EN' and 'CN' for now!")

# Initializing the pipeline
pipeline = IPC_Generation()
task_config = config_params.task_config
prompt_input = {'task_description': task_config.task_description, 'instruction': task_config.instruction, 'batch_size': task_config.batch_size}
generate_prompt = PROMPT_REGISTRY.module_dict['representative_sample'].format_map(prompt_input)
samples = pipeline.run(prompt=generate_prompt)

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