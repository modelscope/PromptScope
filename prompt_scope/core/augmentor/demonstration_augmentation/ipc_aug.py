import concurrent.futures

from loguru import logger
from tqdm import tqdm
from typing import Union, List, Dict, Any
from pydantic import Field, BaseModel

from prompt_scope.core.augmentor.demonstration_augmentation.base_demo_augmention import BaseDemonstrationAugmentation
from prompt_scope.core.llms.base import BaseLLM
from prompt_scope.core.enumeration.language_enum import LanguageEnum


class IPCGeneration(BaseDemonstrationAugmentation):
    """
    The main pipeline for IPC-based demonstration augmentation.
    """
    def __init__(self, 
                 generation_llm: BaseLLM, 
                 language: LanguageEnum, 
                 task_description: str, 
                 instruction: str, 
                 samples_per_step: int, 
                 batch_size: int,
                 workers: int, 
                 **kwargs):
        super().__init__(language, **kwargs)
        self.generation_llm = generation_llm
        self.task_description = task_description
        self.instruction = instruction
        self.samples_per_step = samples_per_step
        self.batch_size = batch_size
        self.workers = workers
        self.FILE_PATH = __file__
    def init_model(self):
        pass

    def init_config(self):
        """
        Initialize the configuration file
        """
        pass

    @staticmethod
    def generate_samples_batch(batch_input, num_samples, batch_size):
        """
        Generate samples in batch, reminders are disgarded.
        """
        batch_num = num_samples // batch_size
        all_batches = [batch_input for _ in range(batch_num)]
        return all_batches

    @staticmethod
    def batch_call(inputs: list[dict], num_workers: int, llm: BaseLLM):
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
            result = llm.chat(messages=prompt)
            pbar.update(1)  # Update the progress bar
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=len(inputs), desc="Processing samples") as pbar:
                all_results = list(executor.map(answer_with_progress, prompt_generator()))

        return all_results

    def run(self, seed_demonstrations: Union[str, List[str], Dict, Any]=[],
            n: int=0, **kwargs) -> List:
        """
        generate samples
        """
        prompt_input = {'task_description': self.task_description,
                        'instruction': self.instruction,
                        'batch_size': self.batch_size}

        prompt = self.prompt_handler.adv_sample_classification.format_map(prompt_input)
        batch_input = prompt
        batch_inputs = self.generate_samples_batch(batch_input, self.samples_per_step,
                                                   self.batch_size)
        samples_batches = self.batch_call(batch_inputs, self.workers, self.generation_llm)
        # if self.module_name == 'dashscope_generation':
        #     try:
        #         samples_lists = [samples_batch.message.content.split("||") for samples_batch in samples_batches]
        #     except Exception:
        #         samples_lists = [samples_batch.output.text.split("||") for samples_batch in samples_batches]
        # elif self.module_name == 'openai_generation' or self.module_name == 'openai_post':
        samples_lists = [samples_batch.message.content.split("||") for samples_batch in samples_batches]

        samples_list = [item.strip() for sample_list in samples_lists for item in sample_list if item]
        logger.info(samples_list)
        return samples_list
    

