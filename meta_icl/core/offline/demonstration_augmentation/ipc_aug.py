import os
from tqdm import tqdm
import concurrent.futures

from meta_icl.core.offline.demonstration_augmentation.base_demo_augmention import BaseDemonstrationAugmentation
from meta_icl.core.utils.logger import Logger
from meta_icl.core.models.generation_model import GenerationModel
from meta_icl.core.utils.utils import load_yaml


class IPC_Generation(BaseDemonstrationAugmentation):
    """
    The main pipeline for IPC-based demonstration augmentation.
    """

    def __init__(self):
        """
        Initialize a new instance of the ClassName class.
        :param config: The configuration file (EasyDict)
        :param task_description: Describe the task that needed to be solved
        :param initial_prompt: Provide an initial prompt to solve the task
        :param output_path: The output dir to save dump, by default the dumps are not saved
        """
        self.init_config()
        self.init_model()
        self.init_prompt()

        self.logger = Logger.get_logger(__name__)

    def init_model(self):
        self.generation_llm = GenerationModel(**self.model_config.generation)

    def init_config(self):
        """
        Initialize the configuration file
        """
        from meta_icl import CONFIG_REGISTRY

        self.task_config = CONFIG_REGISTRY.module_dict['task_config']
        self.model_config = CONFIG_REGISTRY.module_dict['model_config']

    def init_prompt(self):
        from meta_icl import PROMPT_REGISTRY
        PROMPT_REGISTRY.batch_register(load_yaml(
            os.path.join(os.path.dirname(__file__), 'prompt', f'ipc_{self.task_config.language.lower()}.yml')))

    # @staticmethod
    # def log_and_print(logger, message):
    #     print(message)
    #     logger.info(message)

    @staticmethod
    def generate_samples_batch(batch_input, num_samples, batch_size):
        """
        Generate samples in batch, reminders are disgarded.
        """
        batch_num = num_samples // batch_size
        all_batches = [batch_input for _ in range(batch_num)]
        return all_batches

    @staticmethod
    def batch_call(inputs: list[dict], num_workers: int, llm: GenerationModel):
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

    def run(self, prompt: str):
        """
        generate samples
        """
        batch_input = prompt
        batch_inputs = self.generate_samples_batch(batch_input, self.task_config.samples_per_step,
                                                   self.task_config.batch_size)
        samples_batches = self.batch_call(batch_inputs, self.task_config.workers, self.generation_llm)
        print(samples_batches)
        try:
            samples_lists = [samples_batch.message.content.split("||") for samples_batch in samples_batches]
        except:
            samples_lists = [samples_batch.output.text.split("||") for samples_batch in samples_batches]
        samples_list = [item.strip() for sample_list in samples_lists for item in sample_list if item]
        self.logger.info(samples_list)
        return samples_list
