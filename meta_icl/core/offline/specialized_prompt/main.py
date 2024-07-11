from optimization_pipeline import IPC
from meta_icl.core.utils import modify_input_for_ranker, validate_generation_config, override_config
import argparse
import os
from meta_icl.core.offline.specialized_prompt.utils import prompt_rewrite, prompt_evaluation
import json
# General Training Parameters

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_description',
                        default='',
                        required=False, type=str, help='Describing the task')
    parser.add_argument('--prompt',
                        default='',
                        required=False, type=str, help='Prompt to use as initial.')
    parser.add_argument('--load_dump', default='', required=False, type=str, help='In case of loading from checkpoint')
    parser.add_argument('--output_dump', default='dump', required=False, type=str, help='Output to save checkpoints')
    parser.add_argument('--num_ranker_steps', default=3, type=int, help='Number of iterations')
    parser.add_argument('--language', default='Chinese', type=str, help='Language used by the prompt')

    opt = parser.parse_args()

    basic_config_path = os.getenv('CONFIG_FILE_PATH')
    ranker_config_path = os.path.join(basic_config_path, f'ipc_rank_{opt.language.lower()}.yml')
    basic_config_path = os.path.join(basic_config_path, f'ipc_default_{opt.language.lower()}.yml')


    ranker_config_params = override_config(ranker_config_path, config_file=basic_config_path)

    if opt.task_description == '':
        task_description = input("Describe the task: ")
    else:
        task_description = opt.task_description

    if opt.prompt == '':
        initial_prompt = input("Initial prompt: ")
    else:
        initial_prompt = opt.prompt

    ranker_pipeline = IPC(ranker_config_params, output_path=os.path.join(opt.output_dump, 'ranker'))
    if opt.load_dump != '':
        ranker_pipeline.load_state(os.path.join(opt.load_dump, 'ranker'))
        ranker_pipeline.predictor.init_chain(ranker_config_params.dataset.label_schema)

    if (ranker_pipeline.cur_prompt is None) or (ranker_pipeline.task_description is None):
        ranker_mod_prompt, ranker_mod_task_desc = modify_input_for_ranker(ranker_config_params, task_description,
                                                                        initial_prompt)
        ranker_pipeline.cur_prompt = ranker_mod_prompt
        ranker_pipeline.task_description = ranker_mod_task_desc

    # print("Generating ranking prompt")
    # ranking_prompt = ranker_pipeline.run_pipeline(opt.num_ranker_steps)['prompt']
    ranking_prompt = ranker_mod_prompt
    print("Rewriting prompts")
    prompts = prompt_rewrite(initial_prompt)
    evaluation = prompt_evaluation(prompts, ranking_prompt)

    print("Saving")
    with open('results.txt', 'a+') as f:
        f.write(f'{evaluation}\n')

if __name__ == '__main__':
    main()