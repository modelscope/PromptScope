import os
from pathlib import Path
import json

from meta_icl.core.utils.utils import load_yaml
from meta_icl.core.enumeration.language_enum import LanguageEnum
from loguru import logger
from meta_icl import CONFIG_REGISTRY
from meta_icl.core.offline.demonstration_augmentation.ipc_aug import IPCGeneration
from meta_icl.core.utils.utils import get_current_date

current_file_path = Path(__file__)
logger.add(f"{current_file_path.parent}/log/{current_file_path.stem}_{get_current_date()}.log", rotation="10 MB", level="INFO")

basic_config_path = os.path.join(os.path.dirname(__file__), 'ipc_aug_en.yml')

config_params = load_yaml(basic_config_path)
logger.info(config_params)

CONFIG_REGISTRY.batch_register(config_params)

if not hasattr(LanguageEnum, config_params.task_config.language.lower()):
    raise NotImplementedError("Only supports 'en' and 'cn' for now!")

# Initializing the pipeline
pipeline = IPCGeneration(language=config_params.task_config.language.lower())
samples = pipeline.run()

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