from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from enum import Enum
from pathlib import PosixPath

from prompt_scope.core.models.search_algo.nodes import BeamNode, MCTSNode


class SearchAlgoName(str, Enum):
    MCTS = "mcts"
    BEAM_SEARCH = "beam_search"

class BaseSearchAlgo(ABC):
    def __init__(self,
                 task,
                 world_model,
                 action_agent,
                 logger=None,
                 seed=0,
                 print_log=True,
                 test_every_step=True,
                 depth_limit=None,
                 ) -> None:
        self.task = task
        self.world_model = world_model
        self.action_agent = action_agent
        self.states = []
        self.logger = logger
        self.print_log = print_log if logger is not None else False
        self.seed = seed
        self.test_every_step = test_every_step
        self.depth_limit = depth_limit

    
    @abstractmethod
    def before_search(self) -> None:
        pass
    
    @abstractmethod
    def search(self) -> List[BeamNode] | List[MCTSNode]:
        pass
    
    @abstractmethod
    def update_nodes(self) -> Tuple[List[BeamNode] | List[MCTSNode], List[float]]:
        pass

    @abstractmethod
    def after_search(self, store_path: PosixPath) -> Tuple[
        List[List[MCTSNode]] | 
        List[BeamNode], 
        Dict[str, List[List[MCTSNode]] | List[MCTSNode] | MCTSNode] | 
        Dict[str, List[List[BeamNode]] | List[BeamNode] | BeamNode]
        ]:
        pass
    
    def get_states(self):
        return self.states

    def process_all_correct_batch(self):
        self.logger.info(f'\n-----------------------------------------------------')
        self.logger.info('all correct: skip updating cur_prompt')
        self.logger.info(f'\n-----------------------------------------------------\n')
