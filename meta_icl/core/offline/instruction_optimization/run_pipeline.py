from optimization_pipeline import IPC
from meta_icl.core.utils import load_yaml, override_config
import argparse

import os

# General Training Parameters
parser = argparse.ArgumentParser()

parser.add_argument('--basic_config_path', default='/mnt1/yunze.gy/Meta-ICL/conf/ipc_configs', type=str, help='Configuration file path')
parser.add_argument('--batch_config_path', default='',
                    type=str, help='Batch classification configuration file path')
parser.add_argument('--prompt',
                    default='',
                    required=False, type=str, help='Prompt to use as initial.')
parser.add_argument('--task_description',
                    default='',
                    required=False, type=str, help='Describing the task')
parser.add_argument('--load_path', default='', required=False, type=str, help='In case of loading from checkpoint')
parser.add_argument('--output_dump', default='dump', required=False, type=str, help='Output to save checkpoints')
parser.add_argument('--num_steps', default=40, type=int, help='Number of iterations')
parser.add_argument('--language', default='Chinese', type=str, help='Language used by the prompt')

opt = parser.parse_args()

basic_config_path = os.path.join(opt.basic_config_path, f'ipc_default_{opt.language.lower()}.yml')

if opt.batch_config_path == '':
    # load the basic configuration using load_yaml
    config_params = load_yaml(basic_config_path)
else:
    # override the basic configuration with the batch configuration
    config_params = override_config(opt.batch_config_path, config_file=basic_config_path)

# check language
if opt.language.lower() != config_params.language.lower():
    raise ValueError("Language inconsistency!")
if config_params.language.lower() not in ["english", "chinese"]:
    raise NotImplementedError("Only supports 'English' and 'Chinese' for now!")

if opt.task_description == '':
    task_description = input("Describe the task: ")
else:
    task_description = opt.task_description

if opt.prompt == '':
    initial_prompt = input("Initial prompt: ")
else:
    initial_prompt = opt.prompt

# Initializing the pipeline
pipeline = IPC(config_params, task_description, initial_prompt, output_path=opt.output_dump)
if (opt.load_path != ''):
    pipeline.load_state(opt.load_path)
best_prompt = pipeline.run_pipeline(opt.num_steps)
print('\033[92m' + 'Calibrated prompt score:', str(best_prompt['score']) + '\033[0m')
print('\033[92m' + 'Calibrated prompt:', best_prompt['prompt'] + '\033[0m')
