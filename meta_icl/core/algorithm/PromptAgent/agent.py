import os
import time
from datetime import timedelta

from meta_icl.core.algorithm.PromptAgent.utils import get_pacific_time, create_logger
from meta_icl.core.algorithm.PromptAgent.tasks import get_task
from meta_icl.core.algorithm.PromptAgent.search_algo import get_search_algo

from meta_icl.core.utils.logger import Logger
from meta_icl.core.models.base_model import MODEL_REGISTRY
from meta_icl.core.models.generation_model import LlamaIndexGenerationModel
from meta_icl import CONFIG_REGISTRY

class PromptAgent():
    def __init__(self) -> None:
        """
        PromptAgent: set up task, logger, search algorithm, world model
        """
        basic_config = CONFIG_REGISTRY.module_dict["basic_config"]
        task_config = CONFIG_REGISTRY.module_dict["task_config"]
        model_config = CONFIG_REGISTRY.module_dict["model_config"]
        search_config = CONFIG_REGISTRY.module_dict["search_config"]
        world_model_config = CONFIG_REGISTRY.module_dict["world_model_config"]
    
        task_name = basic_config.task_name
        self.task = get_task(task_name)(**task_config)
        self.init_prompt = basic_config.init_prompt
        self.search_algo = basic_config.search_algo
        
        if task_config.get('data_dir', None) and task_name == "bigbench":
            task_name = task_name + "_" + task_config["data_dir"].split('/')[-1].split('.')[-2]
        
        exp_name = f'{get_pacific_time().strftime("%Y%m%d_%H%M%S")}-{task_name}-algo_{self.search_algo}'
        
        self.logger = Logger.get_logger(__name__)
        self.logger.info(exp_name)
        
        
        self.base_model = LlamaIndexGenerationModel(**model_config.base)
        
        self.optim_model = LlamaIndexGenerationModel(**model_config.optim)
        
        self.world_model = MODEL_REGISTRY.module_dict[self.search_algo](
            task=self.task, 
            logger=self.logger, 
            base_model=self.base_model,
            optim_model=self.optim_model, 
            **world_model_config
            )
        
        self.search_algo = get_search_algo(self.search_algo)(
            task=self.task, 
            world_model=self.world_model, 
            logger=self.logger,
            log_dir=basic_config.output_path,
            **search_config
            )
    
    def run(self):
        """
        Start searching from initial prompt
        """
        self.logger.info(f'init_prompt: {self.init_prompt}')
        start_time = time.time()
        
        states, result_dict = self.search_algo.search(init_state=self.init_prompt)
        end_time = time.time()
        exe_time = str(timedelta(seconds=end_time-start_time)).split('.')[0]
        self.logger.info(f'\nDone!Excution time: {exe_time}')
        return states, result_dict

    