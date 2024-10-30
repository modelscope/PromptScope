from __future__ import annotations

import itertools
from typing import Generic, List, Dict, Any, TypeVar
import numpy as np


State = TypeVar("State")
Action = TypeVar("Action")
Trace = tuple[list[State], list[Action]]


class BeamNode(Generic[State, Action]):
    id_iter = itertools.count()
    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self,
                 prompt: str,
                 action: str = None,
                 parent: BeamNode | None = None,
                 ):

        self.id = next(BeamNode.id_iter)
        self.prompt = prompt
        self.test_metric = -1.0
        self.eval_metric = 0.
        self.action = action
        self.parent = parent
        self.children: List[BeamNode] | None = []

        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    def to_dict(self) -> Dict[str, Any]:
        if self.parent is None:
            p_id = -1
        else:
            p_id = self.parent.id

        return {
            'id': self.id,
            'depth': self.depth,
            'parent': p_id,
            'eval_metric': self.eval_metric,
            'test_metric': self.test_metric,
            'prompt': self.prompt,
        }
    
class MCTSNode(Generic[State, Action]):
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self,
                 prompt: str,
                 action: Action | None = None,
                 parent: MCTSNode | None = None,
                 ):
        """
        A node in the MCTS search tree

        :param prompt: the current state
        :param action: the action of the last optimization step,
            i.e., the state transition prompt from parent node to current node
        :param parent: the parent node, None if root of the tree
        """
        self.id = next(MCTSNode.id_iter)

        self.prompt = prompt
        self.action = action
        self.parent = parent
        self.is_terminal = False

        self.children: List[MCTSNode] = []
        self.cum_rewards: list[float] = []
        self.reward = 0.0
        self.test_metric = -1.0
        self.uct = 0.0

        self.visited = 0
        self.expand_times = 0

        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    def calc_q(self, x):
        return np.mean(x)

    def cal_reward(self):
        return self.reward

    @property
    def Q(self) -> float:
        if len(self.cum_rewards) == 0:
            return self.reward
        else:
            return self.calc_q(self.cum_rewards)

    def to_dict(self):
        return {
            'id': self.id,
            'depth': self.depth,
            'parent': -1 if self.parent is None else self.parent.id,
            'visited': self.visited,
            'expand_times': self.expand_times,
            'q': self.Q,
            'uct': self.uct,
            'prompt': self.prompt,
            'reward': self.reward,
            'test_metric': self.test_metric
        }