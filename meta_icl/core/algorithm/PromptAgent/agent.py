import os
import time
from datetime import timedelta

from meta_icl.core.algorithm.PromptAgent.utils import get_pacific_time, create_logger
from meta_icl.core.algorithm.PromptAgent.tasks import get_task
from meta_icl.core.algorithm.PromptAgent.search_algo import get_search_algo
from meta_icl.core.algorithm.base_algorithm import PromptOptimizationWithFeedback
from meta_icl.core.utils.logger import Logger
from meta_icl.core.models.base_model import MODEL_REGISTRY
from meta_icl.core.models.generation_model import LlamaIndexGenerationModel
from meta_icl import CONFIG_REGISTRY

class PromptAgent():
    def __init__(self) -> None:
        """
        PromptAgent: set up task, logger, search algorithm, world model
        """
        self.init_config()
        
        task_name = self.basic_config.task_name
        self.task = get_task(task_name)(**self.task_config)
        self.initial_prompt = self.basic_config.init_prompt
        self.search_algo = self.basic_config.search_algo
        if self.task_config.get('data_dir', None) and task_name == "bigbench":
            task_name = task_name + "_" + self.task_config["data_dir"].split('/')[-1].split('.')[-2]
        exp_name = f'{get_pacific_time().strftime("%Y%m%d_%H%M%S")}-{task_name}-algo_{self.search_algo}'
        self.logger = Logger.get_logger(__name__)
        self.logger.info(exp_name)

        self.init_model()
    
    def init_config(self):
        self.basic_config = CONFIG_REGISTRY.module_dict["basic_config"]
        self.task_config = CONFIG_REGISTRY.module_dict["task_config"]
        self.model_config = CONFIG_REGISTRY.module_dict["model_config"]
        self.search_config = CONFIG_REGISTRY.module_dict["search_config"]
        self.world_model_config = CONFIG_REGISTRY.module_dict["world_model_config"]

    def init_model(self):
        self.base_model = LlamaIndexGenerationModel(**self.model_config.base)
        
        self.optim_model = LlamaIndexGenerationModel(**self.model_config.optim)
        
        self.world_model = MODEL_REGISTRY.module_dict[self.search_algo](
            task=self.task, 
            logger=self.logger, 
            base_model=self.base_model,
            optim_model=self.optim_model, 
            **self.world_model_config
            )
        
        self.search_algo = get_search_algo(self.search_algo)(
            task=self.task, 
            world_model=self.world_model, 
            logger=self.logger,
            log_dir=self.basic_config.output_path,
            **self.search_config
            )
    def run(self):
        """
        set init state, run search algorithm, get best prompt
        """
        self.logger.info(f'init_prompt: {self.initial_prompt}')
        start_time = time.time()

        self.search_algo.before_search(init_state=self.initial_prompt)
        if self.basic_config.search_algo == 'mcts':
            iteration_num = self.search_algo.iteration_num
        elif self.basic_config.search_algo == 'beam_search':
            iteration_num = self.search_algo.depth_limit

        for i in range(iteration_num):
            self.logger.info(
            f'---------------------  iteration {i} ------------------------')
            self.step()
        states, result_dict = self.search_algo.after_search()

        end_time = time.time()
        exe_time = str(timedelta(seconds=end_time-start_time)).split('.')[0]
        self.logger.info(f'\nDone!Excution time: {exe_time}')
        return states, result_dict
    
    def step(self):
        self.logger.info('Searching Path')
        path = self.search_algo.search()
        self.logger.info('Update Prompt According to Error Analysis')
        path, cum_rewards = self.update_with_error(path)
    def update_with_error(self, path):
        return self.search_algo.update_nodes(path)


    