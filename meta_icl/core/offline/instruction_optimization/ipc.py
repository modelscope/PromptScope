from pathlib import Path
import pickle
import os
import json
# import wandb
import random
from typing import Union, List

from meta_icl.core.evaluation.evaluator import Eval
from meta_icl.core.offline.demonstration_augmentation.ipc_aug import IPC_Generation
from meta_icl.core.utils.logger import Logger
from meta_icl.core.utils.utils import load_yaml
from meta_icl.core.utils.prompt_handler import PromptHandler
from meta_icl.core.models.generation_model import GenerationModel
from meta_icl import CONFIG_REGISTRY
from meta_icl.algorithm.base_algorithm import PromptOptimizationWithFeedback

class IPC_Optimization(PromptOptimizationWithFeedback):
    FILE_PATH: str = __file__

    def __init__(self, language: str = "cn", **kwargs):
        super().__init__(language=language, **kwargs)

        self.init_config()
        self.init_model()
        self.patient: int = 0
        self.samples: List[str] = None
        self.logger: Logger = Logger.get_logger(__name__)
        self.cur_step: int = 0
        self.cur_prompt: str = self.task_config.instruction
        self.eval = Eval(FILE_PATH=self.FILE_PATH)

        
    def init_model(self):
        self.generation_llm = GenerationModel(**self.model_config.generation)
        self.predictor_llm = GenerationModel(**self.model_config.predictor)
        self.annotator = GenerationModel(**self.model_config.annotator)

    def init_config(self):
        self.task_config = CONFIG_REGISTRY.module_dict['task_config']
        self.model_config = CONFIG_REGISTRY.module_dict['model_config']
        self.ranker_config = CONFIG_REGISTRY.module_dict.get('ranker_config', None)
        self.eval_config = CONFIG_REGISTRY.module_dict.get('eval_config', None)
        if hasattr(self, 'eval'):
            self.eval.init_config()

    def run(self, **kwargs):
        # Run the optimization pipeline for num_steps
        if kwargs.get('mode', '') == 'ranking':
            self.modify_input_for_ranker()
        if 'ranking_prompt' in kwargs:
            self.eval.eval_instruction = kwargs['ranking_prompt']
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
        if kwargs.get('mode', '') == 'generation':
            prompt_type = 'adv_sample_generation'
        else:
            prompt_type = 'adv_sample_classification'

        mode = kwargs.get('mode', 'classification')

        if not hasattr(kwargs, 'data'):
            if not self.samples:
                self.logger.info('Dataset is empty generating initial samples')
                prompt_input = {'task_description': self.task_config.task_description, 'instruction': self.cur_prompt, 'batch_size': self.task_config.batch_size}
                generate_prompt = getattr(self.prompt_handler, prompt_type).format_map(prompt_input)
                self.samples = self.generate(prompt=generate_prompt)
            else:
                self.logger.info('Generating Adversarials')
                self.step_generate()

        samples = [sample.split('|') for sample in self.samples]
        eval_kwargs = {}
        eval_kwargs['prompt'] = self.cur_prompt

        if kwargs.get('mode', '') == 'generation':
            self.logger.info('Calculating Score and Error Analysis')
            self.eval.init_config()
            eval_kwargs['score'], eval_kwargs['errors'] = self.eval.eval_with_llm(samples=[sample[-1] for sample in samples], prompt_handler=self.prompt_handler)
        else:
            self.logger.info('Running annotator')
            annotations = self.annotate([sample[1] for sample in samples])
            self.logger.info('Running predictor')
            predictions = self.predict([sample[1] for sample in samples])
            self.logger.info('Calculating Score and Error Analysis')
            eval_kwargs['score'], eval_kwargs['corrects'], eval_kwargs['errors'], eval_kwargs['conf_matrix'] = self.eval.eval_accuracy(annotations, predictions, self.task_config.label_schema)
        self.eval.error_analysis(prompt_handler=self.prompt_handler, **eval_kwargs)
        self.logger.info('Updating Prompt')
        self.update_cur_prompt(mode)
        if self.stop_criteria():
            self.logger.info('Stop criteria reached')
            return True
        self.save_state(mode)     
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
            annotate_prompt = self.prompt_handler.annotate.format(samples=sample_str, instruction=self.task_config.instruction, batch_size=self.task_config.batch_size)
            # print('#############\n', annotate_prompt, '################\n')
            response = self.annotator.call(prompt=annotate_prompt)
            try:
                response_list = [item for item in response.message.content.split("||") if item]
            except:
                response_list = [item for item in response.output.text.split("||") if item]
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
            prediction_prompt = self.prompt_handler.predict.format(samples=sample_str, instruction=self.task_config.instruction, batch_size=self.task_config.batch_size)
            # print('#############\n', prediction_prompt, '################\n')
            response = self.annotator.call(prompt=prediction_prompt)
            try:
                response_list = [item for item in response.message.content.split("||") if item]
            except:
                response_list = [item for item in response.output.text.split("||") if item]
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

    def update_cur_prompt(self, mode):
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
        if mode == 'generation':
            generate_prompt = self.prompt_handler.prompt_generation_generation.format_map(prompt_input)
        else:
            generate_prompt = self.prompt_handler.prompt_generation.format_map(prompt_input)
        try:
            prompt_suggestion = self.generation_llm.call(prompt=generate_prompt).message.content
        except:
            prompt_suggestion = self.generation_llm.call(prompt=generate_prompt).output.text
        self.logger.info(prompt_suggestion)
        self.logger.info(f'Previous prompt score:\n{self.eval.mean_score}\n#########\n')
        self.logger.info(f'Get new prompt:\n{prompt_suggestion}')
        self.cur_prompt = prompt_suggestion
    
    def generate(self, prompt: str):
        """
        generate samples
        """
        batch_input = prompt
        batch_inputs = IPC_Generation.generate_samples_batch(batch_input, self.task_config.samples_per_step, self.task_config.batch_size)
        samples_batches = IPC_Generation.batch_call(batch_inputs, self.task_config.workers, self.generation_llm)
        try:
            samples_lists = [samples_batch.message.content.split("||") for samples_batch in samples_batches]
        except:
            samples_lists = [samples_batch.output.text.split("||") for samples_batch in samples_batches]
        samples_list = [item.strip() for sample_list in samples_lists for item in sample_list if item]
        self.logger.info(samples_list)
        return samples_list
    
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
            generate_prompt = self.prompt_handler.step_adv_sample_classification.format_map(prompt_input)
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

    def save_state(self, mode):
        """
        Save the process state
        """
        if not mode:
            return
        if not hasattr(self.task_config, 'output_path') or self.task_config.output_path is None:
            return
        self.logger.info('Save state')
        output_path = os.path.join(self.task_config.output_path, mode)
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
        prompt_mod_prompt = self.prompt_handler.ipc_ranker_prompt_mod.format_map(prompt_input)
        prompt_input = {'task_description': self.task_config.task_description}
        description_mod_prompt = self.prompt_handler.ipc_ranker_description_mod.format_map(prompt_input)

        try:
            mod_prompt = self.generation_llm.call(prompt=prompt_mod_prompt).message.content
            mod_description = self.generation_llm.call(prompt=description_mod_prompt).message.content
        except:
            mod_prompt = self.generation_llm.call(prompt=prompt_mod_prompt).output.text
            mod_description = self.generation_llm.call(prompt=description_mod_prompt).output.text
        self.logger.info(f"Task description modified for ranking to: \n{mod_description}")

        self.logger.info(f"Initial prompt modified for ranking to: \n{mod_prompt}")

        self.task_config.instruction = mod_prompt
        self.task_config.task_description = mod_description