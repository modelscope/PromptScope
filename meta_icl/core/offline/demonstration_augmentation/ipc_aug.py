import os
from tqdm import tqdm
import concurrent.futures
from easydict import EasyDict as edict


from meta_icl.core.utils.logger import Logger
from meta_icl.core.models.generation_model import LlamaIndexGenerationModel
from meta_icl.core.utils.ipc_config import load_yaml
# from meta_icl.core.models.base_model import BaseModel

class IPC_Generation:
    """
    The main pipeline for IPC-based demonstration augmentation.
    """

    def __init__(self, config: edict):
        """
        Initialize a new instance of the ClassName class.
        :param config: The configuration file (EasyDict)
        :param task_description: Describe the task that needed to be solved
        :param initial_prompt: Provide an initial prompt to solve the task
        :param output_path: The output dir to save dump, by default the dumps are not saved
        """
        llm_config = config.model_config.generation
        
        self.llm = LlamaIndexGenerationModel(**llm_config)
        self.logger = Logger.get_logger(__name__)

    @staticmethod
    def log_and_print(logger, message):
        print(message)
        logger.info(message)

    def stop_criteria(self):
        """
        Check if the stop criteria holds. The conditions for stopping:
        1. Usage is above the threshold
        """
        if 0 < self.config.stop_criteria.max_usage < self.calc_usage():
            return True
        return False

    @staticmethod
    def generate_samples_batch(batch_input, num_samples, batch_size):
        """
        Generate samples in batch, reminders are disgarded.
        """
        batch_num = num_samples // batch_size
        all_batches = [batch_input for _ in range(batch_num)]
        return all_batches
    
    @staticmethod
    def batch_call(inputs: list[dict], num_workers: int, llm: LlamaIndexGenerationModel):
        """
        Invoke the chain on a batch of inputs either async or not
        :param inputs: The list of all inputs
        :param num_workers: The number of workers
        :return: A list of results
        """
        def prompt_generator():
            for prompt in inputs:
                yield prompt

        def answer_with_progress(prompt):
            result = llm.call(prompt=prompt)
            pbar.update(1)  # Update the progress bar
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=len(inputs), desc="Processing samples") as pbar:
                all_results = list(executor.map(answer_with_progress, prompt_generator()))

        return all_results
    def generate(self, config: edict):
        """
        generate samples
        """
        task_config = config.task_config

        prompt_template = load_yaml(os.path.join(os.path.dirname(__file__), 'prompt', f'{task_config.language.lower()}.yml'))
        prompt = prompt_template['sample_generation'].format(task_description=task_config.task_description, instruction=task_config.instruction, batch_size=task_config.batch_size)

        batch_input = prompt
        batch_inputs = self.generate_samples_batch(batch_input, task_config.num_samples, task_config.batch_size)
        samples_batches = self.batch_call(batch_inputs, task_config.workers, self.llm)
        samples_lists = [samples_batch.message.content.split("\n\n") for samples_batch in samples_batches]
        samples_list = [item for sample_list in samples_lists for item in sample_list]
        self.log_and_print(self.logger, samples_list)
        return samples_list