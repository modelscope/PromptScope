from abc import ABC, abstractmethod
from typing import List, Union, Any, Dict


class BaseDemoAugmentation(ABC):
    @abstractmethod
    def generate(self, example: Union[str, List[str], Dict], n: int) -> List[str]:
        pass


class SimilarDemoAugmentation(BaseDemoAugmentation):

    def __init__(self):
        pass

    def generate(self, example: Union[str, List[str], Dict], n: int) -> List[str]:
        pass

    def register_prompt(self):
        pass
