import os
import time
from datetime import timedelta
import sys

from meta_icl.algorithm.PromptAgent.utils import get_pacific_time
from meta_icl.algorithm.PromptAgent.tasks import get_task
from meta_icl.algorithm.PromptAgent.search_algo import get_search_algo
from meta_icl.algorithm.base_algorithm import PromptOptimizationWithFeedback
from loguru import logger
from meta_icl.core.models.base_model import MODEL_REGISTRY
from meta_icl.core.models.generation_model import GenerationModel, OpenAIGenerationModel, OpenAIPostModel
from meta_icl import CONFIG_REGISTRY
from meta_icl.core.utils.utils import load_yaml
from meta_icl.core.enumeration.language_enum import LanguageEnum

class PromptAgent(PromptOptimizationWithFeedback):
    """
    PromptAgent (PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization) 
    is designed to optimize prompts for language models, by initializing tasks, configuring settings, 
    selecting search algorithms and models, logging progress, and determining the most effective prompt through iterative refinement.
    """
    FILE_PATH: str = __file__
    def __init__(self, 
                 dataset_path: str = __file__,
                 language: LanguageEnum = "en",
                 **kwargs
                 ) -> None:
        """
        Initialize the PromptAgent with necessary configurations, task setup, search algorithm selection,
        and logging initialization. The agent then readies the model for the prompt optimization process.

        Args:
            dataset_path (str): Path to the dataset used for the task.
            language (LanguageEnum): Language setting for the tasks and prompts.
            **kwargs: Additional keyword arguments for flexibility in configurations.

        Post Init:
            - Configurations are initialized and potentially updated based on kwargs.
            - A task instance is created according to the specified task name and configuration.
            - An appropriate search algorithm is set up based on the chosen strategy.
            - A logging mechanism is configured to track the experiment under a named experiment ID.
            - The AI model is initialized in preparation for prompt evaluations and optimizations.
        """
        super().__init__(language=language, **kwargs)
        self.init_config()
        
        task_name = self.basic_config.task_name
        self.dataset_path = dataset_path
        self.task_config.data_dir = self.dataset_path
        self.task = get_task(task_name)(**self.task_config)
        self.initial_prompt = self.basic_config.init_prompt
        self.search_algo = self.basic_config.search_algo
        if self.task_config.get('data_dir', None) and task_name == "bigbench":
            task_name = task_name + "_" + self.task_config["data_dir"].split('/')[-1].split('.')[-2]
        exp_name = f'{get_pacific_time().strftime("%Y%m%d_%H%M%S")}-{task_name}-algo_{self.search_algo}'
        logger.info(exp_name)

        self.init_model()

    def init_config(self):
        """
        Initializes the configuration for the agent by retrieving various configuration sections
        from the global CONFIG_REGISTRY. This sets up the basic, task-specific, model, search,
        and world model configurations required for the agent's operation.

        The configurations are retrieved from a registry module which centralizes the access to 
        different parts of the setup, ensuring modularity and easier management of settings.
        """
        self.basic_config = CONFIG_REGISTRY.module_dict["basic_config"]
        self.task_config = CONFIG_REGISTRY.module_dict["task_config"]
        self.model_config = CONFIG_REGISTRY.module_dict["model_config"]
        self.search_config = CONFIG_REGISTRY.module_dict["search_config"]
        self.world_model_config = CONFIG_REGISTRY.module_dict["world_model_config"]

    def init_model(self):
        """
        Initializes the models and search algorithm for the prompt optimization process.

        This method sets up the base model, optimization model, world model, and search algorithm
        based on the configurations provided. It also initializes the logging mechanism for tracking
        the optimization process.
        """
        base_module_name = self.model_config.base.get('module_name')
        if base_module_name == 'dashscope_generation':
            self.base_model = GenerationModel(**self.model_config.base)
        elif base_module_name == 'openai_generation':
            self.base_model = OpenAIGenerationModel(**self.model_config.base)
        elif base_module_name == 'openai_post':
            self.base_model = OpenAIPostModel(**self.model_config.base)

        optim_module_name = self.model_config.optim.get('module_name')
        if optim_module_name == 'dashscope_generation':
            self.optim_model = GenerationModel(**self.model_config.optim)
        elif optim_module_name == 'openai_generation':
            self.optim_model = OpenAIGenerationModel(**self.model_config.optim)
        elif optim_module_name == 'openai_post':
            self.optim_model = OpenAIPostModel(**self.model_config.optim)
                    
        self.world_model = MODEL_REGISTRY.module_dict[self.search_algo](
            task=self.task, 
            logger=logger, 
            base_model=self.base_model,
            optim_model=self.optim_model,
            prompt_handler=self.prompt_handler, 
            **self.world_model_config
            )
        
        self.search_algo = get_search_algo(self.search_algo)(
            task=self.task, 
            world_model=self.world_model, 
            logger=logger,
            log_dir=self.basic_config.output_path,
            **self.search_config
            )
    def run(self):
        """
        Orchestrates the search for the best prompt by initializing the search with a given prompt,
        executing iterations as per the configured search algorithm, logging each step, measuring
        total execution time, and extracting the optimal prompt along with related metadata.

        The specific number of iterations is determined by the configuration of the search algorithm
        ('mcts' uses `iteration_num`, while 'beam_search' refers to `depth_limit`).

        Returns:
            tuple: A pair containing:
                - states: Relevant states or information from the search process leading to the best prompt.
                - result_dict: A dictionary encapsulating details about the best prompt and potentially its score or metrics.
        """
        logger.info(f'init_prompt: {self.initial_prompt}')
        start_time = time.time()

        self.search_algo.before_search(init_state=self.initial_prompt)
        if self.basic_config.search_algo == 'mcts':
            iteration_num = self.search_algo.iteration_num
        elif self.basic_config.search_algo == 'beam_search':
            iteration_num = self.search_algo.depth_limit

        for i in range(iteration_num):
            logger.info(
            f'---------------------  iteration {i} ------------------------')
            self.step()

        states, result_dict = self.extract_best_prompt()
        end_time = time.time()
        exe_time = str(timedelta(seconds=end_time-start_time)).split('.')[0]
        logger.info(f'\nDone!Excution time: {exe_time}')
        return states, result_dict

    def step(self):
        """
        Executes a single step in the optimization process which includes searching for the best path 
        and updating prompts based on error analysis.
        """
        logger.info('Searching Path')
        path = self.search_algo.search() # ⭐ Conducts a search for the optimal path
        logger.info('Update Prompt According to Error Analysis')
        path, cum_rewards = self.update_with_error(path) # ⭐ Refines prompts along the found path using error feedback
    def update_with_error(self, path):
        """
        Updates nodes within the search algorithm's data structure based on the error from the executed path,
        effectively refining the prompts.

        Args:
            path: The sequence of nodes representing the path chosen for update.

        Returns:
          A tuple containing the updated path and cumulative rewards from this update step.
        """
        return self.search_algo.update_nodes(path)
    
    def extract_best_prompt(self):
        """
        Retrieves the best prompt discovered by the search algorithm after completing the search process.

        Returns:
            The most effective prompt generated by the optimization, intended to enhance AI system interactions.
        """
        return self.search_algo.after_search()


    