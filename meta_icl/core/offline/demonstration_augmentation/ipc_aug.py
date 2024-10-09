"""
todo: by jm, unify the interface design of argumentation method.
1. support the input of the demonstration except loading from the configuration.
2. support the parameters loading from the init function rather than configuration dict?
3. What is the difference between the similar argumentation and ipc argumentation?
4. Little confused by the prompt in ipc_aug.yaml, which is more like a initial proposal prompt?
"""
import concurrent.futures

from loguru import logger
from tqdm import tqdm

from meta_icl.core.models.generation_model import GenerationModel, OpenAIGenerationModel, OpenAIPostModel
from meta_icl.core.offline.demonstration_augmentation.base_demo_augmention import BaseDemonstrationAugmentation


class IPCGeneration(BaseDemonstrationAugmentation):
    """
    The main pipeline for IPC-based demonstration augmentation.
    """
    FILE_PATH: str = __file__

    def __init__(self, language: str = "cn", **kwargs):
        """
        Initialize a new instance of the ClassName class.
        :param language: prompt language
        """
        super().__init__(language=language, **kwargs)
        self.init_config()
        self.init_model()

    def init_model(self):
        self.module_name = self.model_config.generation.get('module_name')
        if self.module_name == 'dashscope_generation':
            self.generation_llm = GenerationModel(**self.model_config.generation)
        elif self.module_name == 'openai_generation':
            self.generation_llm = OpenAIGenerationModel(**self.model_config.generation)
        elif self.module_name == 'openai_post':
            self.generation_llm = OpenAIPostModel(**self.model_config.generation)

    def init_config(self):
        """
        Initialize the configuration file
        """
        from meta_icl import CONFIG_REGISTRY

        self.task_config = CONFIG_REGISTRY.module_dict['task_config']
        self.model_config = CONFIG_REGISTRY.module_dict['model_config']

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

    def run(self, prompt: str = ""):
        """
        generate samples
        """
        # todo: by zy, support the input of the demonstration except loading from the configuration.
        prompt_input = {'task_description': self.task_config.task_description,
                        'instruction': self.task_config.instruction,
                        'batch_size': self.task_config.batch_size}

        if not prompt:
            prompt = self.prompt_handler.representative_sample.format_map(prompt_input)
        batch_input = prompt
        batch_inputs = self.generate_samples_batch(batch_input, self.task_config.samples_per_step,
                                                   self.task_config.batch_size)
        samples_batches = self.batch_call(batch_inputs, self.task_config.workers, self.generation_llm)
        if self.module_name == 'dashscope_generation':
            try:
                samples_lists = [samples_batch.message.content.split("||") for samples_batch in samples_batches]
            except:
                samples_lists = [samples_batch.output.text.split("||") for samples_batch in samples_batches]
        elif self.module_name == 'openai_generation' or self.module_name == 'openai_post':
            samples_lists = [samples_batch.message.content.split("||") for samples_batch in samples_batches]

        samples_list = [item.strip() for sample_list in samples_lists for item in sample_list if item]
        logger.info(samples_list)
        return samples_list
