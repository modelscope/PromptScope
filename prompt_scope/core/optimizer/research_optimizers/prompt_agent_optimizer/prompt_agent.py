from loguru import logger
from typing import Literal, Dict, List, Tuple
from pydantic import Field, model_validator, ConfigDict
from pathlib import Path, PosixPath
import os
from datetime import datetime

from prompt_scope.core.models.search_algo.base_algo import BaseSearchAlgo
from prompt_scope.core.optimizer.research_optimizers.prompt_agent_optimizer.tasks.base_task import BaseTask
from prompt_scope.core.optimizer.research_optimizers.base_algorithm import PromptOptimizationWithFeedback
from prompt_scope.core.models.search_algo.beam_search import BeamNode
from prompt_scope.core.models.search_algo.mcts import MCTSNode

def create_dated_directory(base_path):
    # Get current date in YYYYMMDD format
    current_date = datetime.now().strftime("%Y%m%d")
    version = 1
    
    # Keep incrementing version until we find a directory that doesn't exist
    while True:
        dir_name = f"{current_date}_v{version}"
        full_path = os.path.join(base_path, dir_name)
        
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            return full_path
        
        version += 1

class PromptAgent(PromptOptimizationWithFeedback):
    """
    PromptAgent (PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization)
    is designed to optimize prompts for language models, by initializing tasks, configuring settings,
    selecting search algorithms and models, logging progress, and determining the most effective prompt through iterative refinement.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed = True
    )
    # =============LLM and model Configuration=============
    search_algo: BaseSearchAlgo = Field(..., description="Search Algorithm.")

    # =============Path Configuration=============
    dataset_name: Literal["bbh"] = Field(default="bbh", description="Name of the dataset")
    task: BaseTask = Field(..., description="task for the experiment")
    task_name: str = Field(default="bigbench", description="task name for the experiment")
    prompt_path: str = Field(default=__file__, description="Prompt file path")
    store_path: str = Field(default=Path(__file__).parent.joinpath("pagent_output"))

    # =============Experiment Configuration=============
    search_algo_name: Literal["mcts", "beam_search"] = Field(...)
    world_model_name: Literal["base", "beam_search"] = Field(...)
    verbose: bool = True
    train_size: int = 4
    eval_size: int = Field(default=3, description="data split for reward calculation")
    test_size: int = Field(default=3, description="if test_size is not 0, the optimized nodes will be tested at last.")
    seed: int = Field(default=42, description="if need to fixed shuffled dataset.")
    post_instruction: bool = Field(default=False, description="false: prompt + task question | true: task question + prompt")

    # =============Search Configuration=============
    iteration_num: int = 2
    expand_width: int = Field(default=2, description="num of branches of each node")
    depth_limit: int = Field(default=2, description="the max depth of mcts")
    # mcts setting
    min_depth: int | None = Field(default=None, examples=[2], description="min depth of mcts for early stop")
    w_exp: float | None = Field(default=None, examples=[2.5], description="balance exploration and exploitation")
    # beam search setting
    beam_width: int | None = Field(default=None, examples=[2])

    # =============World Model Configuration=============
    # mcts world model setting
    train_shuffle: bool = True
    num_new_prompts: int = 1 # 3 if beam search
    train_batch_size: int = 5
    prompt_length_limit: int = 200


    @model_validator(mode='before')
    def validate(cls, data: Dict) -> Dict:
        """Check the configuration of the algorithm"""
        if data["search_algo_name"] == "mcts":
            assert data["world_model_name"] == "base"
        elif data["search_algo_name"] == "beam_search":
            assert data["world_model_name"] == "beam_search"
        return data


    def _before_run(self) -> None:
        logger.info(f'init_prompt: {self.instruction}')
        self.store_path = create_dated_directory(self.store_path)
        self.search_algo.before_search(init_state=self.instruction)
        if self.search_algo == 'mcts':
            self.num_step = self.iteration_num
        elif self.search_algo == 'beam_search':
            self.num_step = self.depth_limit


    def _after_run(self):
        best_prompt = self.extract_best_prompt()
        return best_prompt


    def _step(self, i_step) -> bool:
        """
        Executes a single step in the optimization process which includes searching for the best path
        and updating prompts based on error analysis.
        """
        logger.info('Searching Path')
        path = self._predict(i_step=i_step)  # ⭐ Conducts a search for the optimal path
        logger.info('Update Prompt According to Trajectory Prompts')
        path, cum_rewards = self._update_prompt(path=path)  # ⭐ Refines prompts along the found path using error feedback
        ## TODO: implement early stop
        return False

    def _predict(self, **kwargs) -> List[BeamNode] | List[MCTSNode]:
        return self.search_algo.search(**kwargs)


    def _evaluate_and_analyze(self):
        """
        TODO: included in the self.search_algo.
        """
        raise NotImplementedError


    def _update_prompt(self, *, path: List) -> Tuple[List[BeamNode] | List[MCTSNode], List[float]]:
        """
        Updates nodes within the search algorithm's data structure based on the error from the executed path,
        effectively refining the prompts.

        Args:
            path: The sequence of nodes representing the path chosen for update.

        Returns:
          A tuple containing the updated path and cumulative rewards from this update step.
        """
        return self.search_algo.update_nodes(path)
        
        

    def extract_best_prompt(self) -> str:
        """
        Retrieves the best prompt discovered by the search algorithm after completing the search process.

        Returns:
            The most effective prompt generated by the optimization, intended to enhance AI system interactions.
        """
        _, output = self.search_algo.after_search(store_path=self.store_path)
        if self.search_algo_name == 'mcts':
            best_node = output.get("best_reward_path_selected_node", [MCTSNode(prompt="")])[0]
        elif self.search_algo_name == 'beam_search':
            best_node = output.get("best_global_node", [BeamNode(prompt="")])[0]
        return best_node.prompt
