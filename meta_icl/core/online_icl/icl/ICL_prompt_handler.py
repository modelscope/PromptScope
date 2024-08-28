from meta_icl.core.utils.prompt_handler import PromptHandler
from typing import Union, List, Dict
from meta_icl.core.enumeration.language_enum import LanguageEnum
import re
from loguru import logger


class ICLPromptHandler(PromptHandler):
    def __init__(self,
                 class_path: str,
                 language: Union[LanguageEnum, str],
                 class_name: str = "",
                 prompt_file: str = "",
                 prompt_dict: dict = None,
                 **kwargs):
        super().__init__(class_path, language, class_name, prompt_file, prompt_dict, **kwargs)

    @staticmethod
    def get_var_name_in_str(template: str):
        variables = re.findall(r'\{(\w+)\}', template)
        return variables

    @staticmethod
    def merge_dict(dict_list: List[Dict]):
        merged_dict = {}
        for item in dict_list:
            if isinstance(item, dict):
                for k, v in item.items():
                    merged_dict[k] = v
        return merged_dict

    def template_filed_check(self, template: str, fill_dict: dict):
        template_var_list = self.get_var_name_in_str(template)
        pass_check = True
        for var in template_var_list:
            if var not in fill_dict.keys():
                logger.error(f"{var} is not in fill_dict: {fill_dict}")
                pass_check = False
                return pass_check
        return pass_check

    def get_prompt_demo_str(self, retrieved_examples: list) -> str:
        example_template = self.prompt_dict.get('example_template')
        print(retrieved_examples[0].keys())
        assert self.template_filed_check(example_template, retrieved_examples[0])
        return '\n'.join(example_template.format_map(example) for example in retrieved_examples)

    def organize_icl_prompt(self, selection_examples, cur_query, configs: dict = None) -> str:
        """

        """
        icl_template = self.prompt_dict.get('icl_template')
        logger.info(f"icl_template: {icl_template}")
        instruction = self.prompt_dict.get('instruction')
        query_template = self.prompt_dict.get('query_template')
        # fill in variables in the instruction
        instruction = instruction.format_map(configs)
        logger.info(f"instruction: {instruction}")

        # fill in retrieved examples into the example_template
        examples = self.get_prompt_demo_str(selection_examples)
        logger.info(f"examples: {examples}")

        # fill in cur_query into query_template
        query_str = query_template.format_map(self.merge_dict([cur_query, configs]))
        logger.info(f"query_str: {query_str}")

        return icl_template.format(instruction=instruction, examples=examples, query_str=query_str)