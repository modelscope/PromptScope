from pathlib import Path
import pickle
import os
import json
# import wandb
import random

from meta_icl.core.evaluation.evaluator import Eval
from meta_icl.core.offline.demonstration_augmentation.ipc_aug import IPC_Generation
from meta_icl.core.utils.logger import Logger
from meta_icl.core.utils.ipc_config import load_yaml
from meta_icl.core.models.generation_model import LlamaIndexGenerationModel
from meta_icl import CONFIG_REGISTRY, PROMPT_REGISTRY

class IPC_Optimization(IPC_Generation):
    """
    The main pipeline for intent-based prompt calibration (IPC). The pipeline is composed of 4 main components:
    1. dataset - The dataset handle the data including the annotation and the prediction
    2. annotator - The annotator is responsible generate the GT
    3. predictor - The predictor is responsible to generate the prediction
    4. eval - The eval is responsible to calculate the score and the large errors
    """

    def __init__(self):
        """
        Initialize a new instance of the ClassName class.
        :param config: The configuration file (EasyDict)
        :param task_description: Describe the task that needed to be solved
        :param initial_prompt: Provide an initial prompt to solve the task
        :param output_path: The output dir to save dump, by default the dumps are not saved
        """
        super().__init__()
        # self.task_config = CONFIG_REGISTRY.module_dict['task_config']
        # self.model_config = CONFIG_REGISTRY.module_dict['model_config']
        # self.global_config = CONFIG_REGISTRY.module_dict['global_config']
        self.ranker_config = CONFIG_REGISTRY.module_dict.get('ranker_config', None)
        self.eval_config = CONFIG_REGISTRY.module_dict.get('eval_config', None)

        self.generation_llm = LlamaIndexGenerationModel(**self.model_config.generation)
        self.predictor_llm = LlamaIndexGenerationModel(**self.model_config.predictor)
        self.annotator = LlamaIndexGenerationModel(**self.model_config.annotator)
        self.prompt_register()

        self.patient = 0
        self.samples = None
        self.logger = Logger.get_logger(__name__)
        self.cur_step = 0
        self.cur_prompt = self.task_config.instruction
        self.eval = Eval()

    def prompt_register(self):
        PROMPT_REGISTRY.batch_register(load_yaml(os.path.join(os.path.dirname(__file__), 'prompt', f'{self.task_config.language.lower()}.yml')))

    def run_pipeline(self, **kwargs):
        # Run the optimization pipeline for num_steps
        if kwargs.get('mode', '') == 'ranking':
            self.modify_input_for_ranker()
        for _ in range(self.task_config.num_steps):
            stop_criteria = self.step(**kwargs)
            if stop_criteria:
                break
        final_result = self.extract_best_prompt()
        return final_result
    
    def step(self, **kwargs):
        """
        This is the main optimization process step.
        """
        self.logger.info(f'Starting step {self.cur_step}')
        if not hasattr(kwargs, 'data'):
            if not self.samples:
                self.logger.info('Dataset is empty generating initial samples')
                prompt_input = {'task_description': self.task_config.task_description, 'instruction': self.cur_prompt, 'batch_size': self.task_config.batch_size}
                generate_prompt = PROMPT_REGISTRY.module_dict['adv_sample_classification'].format_map(prompt_input)
                self.samples = self.generate(prompt=generate_prompt)
            else:
                self.logger.info('Generating Adversarials')
                self.step_generate()

        samples = [sample.split('|')[-1] for sample in self.samples]

        self.logger.info('Running annotator')
        annotations = self.annotate(samples)
        self.logger.info('Running predictor')
        predictions = self.predict(samples)
        self.logger.info('Calculating Score and Error Analysis')
        if kwargs.get('mode', '') == 'generation':
            self.mean_score, self.corrects, self.errors, self.conf_matrix
        else:
            self.mean_score, self.corrects, self.errors, self.conf_matrix = self.eval.eval_accuracy(annotations, predictions, self.task_config.label_schema)
        self.eval.error_analysis(self.cur_prompt, annotations, predictions)
        self.logger.info('Updating Prompt')
        self.update_cur_prompt()
        if self.stop_criteria():
            self.logger.info('Stop criteria reached')
            return True
        self.save_state()       
        self.cur_step += 1
        return False

    def annotate(self, samples: list[str]):
        """
        annotate samples with LLM or argilla
        """
        samples_batches = [samples[i:i + self.task_config.batch_size] for i in range(0, len(samples), self.task_config.batch_size)]
        batch, annotations = 0, []
        for sample_batch in samples_batches:
            sample_str = "|".join(sample_batch)
            annotate_prompt = PROMPT_REGISTRY.module_dict['annotate'].format(samples=sample_str, instruction=self.task_config.instruction, batch_size=self.task_config.batch_size)
            # print('#############\n', annotate_prompt, '################\n')
            response = self.annotator.call(prompt=annotate_prompt)
            response_list = [item for item in response.message.content.split("||") if item]
            # print('#############\n', response_list, '################\n')
            annotations.extend([{"ID": f"{batch}_{lines[0].strip()}", "问题": lines[1], "标注": lines[-1]} for lines in (sample.split('|') for sample in response_list if sample)])
            batch += 1
        self.logger.info(annotations)
        return annotations
    
    def predict(self, samples: list[str]):
        """
        generate samples
        """
        samples_batches = [samples[i:i + self.task_config.batch_size] for i in range(0, len(samples), self.task_config.batch_size)]
        batch, predictions = 0, []
        for sample_batch in samples_batches:
            sample_str = "|".join(sample_batch)
            prediction_prompt = PROMPT_REGISTRY.module_dict['predict'].format(samples=sample_str, instruction=self.task_config.instruction, batch_size=self.task_config.batch_size)
            # print('#############\n', prediction_prompt, '################\n')
            response = self.annotator.call(prompt=prediction_prompt)
            response_list = [item for item in response.message.content.split("||") if item]
            # print('#############\n', response_list, '################\n')
            predictions.extend([{"ID": f"{batch}_{lines[0].strip()}", "问题": lines[1], "预测": lines[-1]} for lines in (sample.split('|') for sample in response_list if sample)])
            batch += 1
        self.logger.info(predictions)
        return predictions

    def extract_best_prompt(self):
        sorted_history = sorted(
            self.eval.history[min(self.task_config.warmup - 1, len(self.eval.history) - 1):],
            key=lambda x: x['score'],
            reverse=False)
        return {'prompt': sorted_history[-1]['prompt'], 'score': sorted_history[-1]['score']}

    def update_cur_prompt(self):
        """
        Run the meta-prompts and get new prompt suggestion, estimated prompt score and a set of challenging samples for the new prompts
        """
        step_num = len(self.eval.history)
        if (step_num < self.task_config.warmup) or (step_num % 3) > 0:
            self.last_history = self.eval.history[-self.task_config.history_length:]
        else:
            sorted_history = sorted(self.eval.history[self.task_config.warmup - 1:], key=lambda x: x['score'],
                                    reverse=False)
            self.last_history = sorted_history[-self.task_config.history_length:]
        history_prompt = '\n'.join([self.eval.sample_to_text(sample) for sample in self.last_history])
        prompt_input = {"history": history_prompt, "task_description": self.task_config.task_description,
                        'error_analysis': self.last_history[-1]['analysis']}
        if hasattr(self.task_config, 'label_schema'):
            prompt_input["labels"] = json.dumps(self.task_config.label_schema)
        generate_prompt = PROMPT_REGISTRY.module_dict['prompt_generation'].format_map(prompt_input)
        prompt_suggestion = self.generation_llm.call(prompt=generate_prompt).message.content
        self.logger.info(prompt_suggestion)
        self.logger.info(f'Previous prompt score:\n{self.eval.mean_score}\n#########\n')
        self.logger.info(f'Get new prompt:\n{prompt_suggestion}')
        self.cur_prompt = prompt_suggestion
    
    def step_generate(self):
        """
        generate new samples with new prompts
        """
        if len(self.samples) < self.task_config.max_samples:
            prompt_input = {'task_description': self.task_config.task_description, 'prompt': self.cur_prompt, 'batch_size': self.task_config.batch_size}
            random.shuffle(self.samples)
            txt_res = '##\n'
            for sample in self.samples[:self.task_config.num_extra_sample]:
                txt_res += f"Sample:\n {sample}\n#\n"
            prompt_input['extra_samples'] = txt_res
            if all([all(len(error_label) == 0 for error_label in t['errors']) for t in self.last_history]): 
                history_samples = '\n'.join([self.eval.sample_to_text(sample,
                                                                 num_errors_per_label=self.task_config.num_errors_per_label,
                                                                 is_score=False) for sample in self.last_history])
                prompt_input['history'] = history_samples
            else:
                prompt_input['history'] = 'No previous errors information'
            generate_prompt = PROMPT_REGISTRY.module_dict['step_adv_sample_classification'].format_map(prompt_input)
            new_samples = self.generate(prompt=generate_prompt)
            self.samples.extend(new_samples)
            self.logger.info(self.samples)
    
    def stop_criteria(self):
        """
        Check if the stop criteria holds. The conditions for stopping:
        1. Usage is above the threshold
        2. There was no improvement in the last > patient steps
        """
        # if 0 < self.config.stop_criteria.max_usage < self.calc_usage():
        #     return True
        if len(self.eval.history) <= self.task_config.warmup:
            self.patient = 0
            return False
        
        if len(self.eval.history) == 1:
            max_score = 0
        else:
            max_score = self.eval.get_max_score(self.task_config.warmup)

        if max_score - self.eval.history[-1]['score'] > -self.task_config.min_delta:
            self.patient += 1
        else:
            self.patient = 0

        if self.patient > self.task_config.patience:
            return True
        return False

    def save_state(self):
        """
        Save the process state
        """
        if not hasattr(self.task_config, 'output_path') or self.task_config.output_path is None:
            return
            
        self.logger.info('Save state')
        output_path = self.task_config.output_path
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        output_path = Path(output_path)

        state = {'history': self.eval.history, 'step': self.cur_step,
                 'prompt': self.cur_prompt, 'task_description': self.task_config.task_description,
                 'patient': self.patient}
        pickle.dump(state, open(output_path / 'history.pkl', 'wb'))

    def load_state(self, path: str):
        """
        Load pretrain state
        """
        path = Path(path)
        if (path / 'history.pkl').is_file():
            state = pickle.load(open(path / 'history.pkl', 'rb'))
            self.eval.history = state['history']
            self.batch_id = state['batch_id']
            self.cur_prompt = state['prompt']
            self.task_description = state['task_description']
            self.patient = state['patient']

    def modify_input_for_ranker(self):
        prompt_input = {'label_schema': self.task_config.label_schema,
                        'prompt': self.task_config.instruction}
        prompt_mod_prompt = PROMPT_REGISTRY.module_dict['ranker_prompt_mod'].format_map(prompt_input)
        prompt_input = {'task_description': self.task_config.task_description}
        description_mod_prompt = PROMPT_REGISTRY.module_dict['ranker_description_mod'].format_map(prompt_input)

        mod_prompt = self.generation_llm.call(prompt=prompt_mod_prompt).message.content
        mod_description = self.generation_llm.call(prompt=description_mod_prompt).message.content
        self.logger.info(f"Task description modified for ranking to: \n{mod_description}")

        self.logger.info(f"Initial prompt modified for ranking to: \n{mod_prompt}")

        self.task_config.instruction = mod_prompt
        self.task_config.task_description = mod_description

    

    
