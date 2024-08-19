from abc import ABC, abstractmethod
from typing import List, Union, Any, Dict


class BaseDemonstrationAugmentation(ABC):
    """
    Base Abstract Class for Prompt Optimization with Feedback
    """

    def __init__(self):
        pass

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def init_config(self):
        pass

    @abstractmethod
    def init_prompt(self):
        pass

    @abstractmethod
    def run(self, seed_demonstrations: Union[str, List[str], Dict, Any],
            n: int, **kwargs) -> List:
        pass
