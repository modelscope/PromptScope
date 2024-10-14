from typing import List, Union, Dict

from meta_icl.core.offline.demonstration_augmentation.base_demo_augmention import BaseDemonstrationAugmentation
# from meta_icl import CONFIG_REGISTRY
from meta_icl.core.utils.demontration_utils import demo_augmentation_by_llm_prompt_org
from meta_icl.core.utils.sys_prompt_utils import call_llm_with_message
from meta_icl.core.utils.demontration_utils import extract_from_markdown_json
from loguru import logger


class SimilarDemoAugmentation(BaseDemonstrationAugmentation):

    def __init__(self, aug_config: Dict, **kwargs):
        super().__init__()
        self.augmentation_config = aug_config

    def run(self,
                 seed_demonstration: Union[str, List[str], Dict, List[Dict]],
                 n: int) -> List:
        # todo: by jm, implement the run function and return the generate results.
        aug_query_prompt = self.formatting_generation_prompt(seed_demonstration, n)
        res = call_llm_with_message(aug_query_prompt)
        logger.info(f"model response: {res}")
        gen_results_list = extract_from_markdown_json(res)
        logger.info(f"generate results: {gen_results_list}")
        return res

    def register_prompt(self):
        pass

    def formatting_generation_prompt(self,
                                     seed_demonstration: Union[str, List[str], Dict, List[Dict]],
                                     n: int):
        """
            Generates formatted prompts based on the provided seed demonstration and generation configuration.

            :param seed_demonstration: Union[str, List[str], Dict, List[Dict]]
                The initial seed demonstration, which can be a string, list of strings, dictionary, or list of dictionaries.
            :param n: int
                The number of examples to generate.
            :return:
                Formatted prompts generated from the seed demonstration.
        """

        generation_prompt = demo_augmentation_by_llm_prompt_org(
            demonstration_text=seed_demonstration,
            demonstration_generation_instruction=self.augmentation_config.get('demonstration_generation_instruction'),
            num_generated_examples=n,
            demonstration_requirements=self.augmentation_config.get('demonstration_requirements', "")
        )
        return generation_prompt
