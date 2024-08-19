import json
from typing import List, Union, Any, Dict
# from meta_icl import CONFIG_REGISTRY
from meta_icl.core.utils.demontration_utils import generate_similar_demonstration, demo_augmentation_by_llm_prompt_org
from meta_icl.core.offline.demonstration_augmentation.base_demo_augmention import BaseDemonstrationAugmentation
from meta_icl.core.utils.sys_prompt_utils import call_llm_with_message
from loguru import logger


class SimilarDemoAugmentation(BaseDemonstrationAugmentation):

    def __init__(self, aug_config: Dict, **kwargs):
        self.augmentation_config = aug_config

    def generate(self,
                 seed_demonstration: Union[str, List[str], Dict, List[Dict]],
                 n: int) -> List:
        aug_query_prompt = self.formatting_generation_prompt(seed_demonstration, n)
        res = call_llm_with_message()

    def register_prompt(self):
        pass

    def formatting_generation_prompt(self,
                                     seed_demonstration: Union[str, List[str], Dict, List[Dict]],
                                     n: int):
        generation_prompt = demo_augmentation_by_llm_prompt_org(
            demonstration_text=seed_demonstration,
            demonstration_generation_instruction=self.augmentation_config.get('demonstration_generation_instruction'),
            num_generated_examples=n,
            demonstration_requirements=self.augmentation_config.get('demonstration_requirements', "")
        )
        return generation_prompt



