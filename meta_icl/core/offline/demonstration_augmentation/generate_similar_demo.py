from abc import ABC, abstractmethod
from typing import List, Union, Any, Dict
from meta_icl import CONFIG_REGISTRY
from meta_icl.core.utils.demontration_utils import generate_similar_demonstration

class BaseDemoAugmentation(ABC):
    @abstractmethod
    def generate(self, seed_example: Union[str, List[str], Dict], n: int) -> List:
        pass


class SimilarDemoAugmentation(BaseDemoAugmentation):

    def __init__(self):
        pass

    def generate(self, seed_example: Union[str, List[str], Dict], n: int) -> List:
        pass

    def register_prompt(self):
        pass
