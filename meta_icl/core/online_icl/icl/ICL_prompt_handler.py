from meta_icl.core.utils.prompt_handler import PromptHandler
from typing import Union, List, Dict
from meta_icl.core.enumeration.language_enum import LanguageEnum
import re
from loguru import logger


class ICLPromptHandler(PromptHandler):
    """
    A class that handles In-Context Learning (ICL) prompts.

    This class extends `PromptHandler` to generate prompts specific to ICL tasks, including creating prompts based on templates and data instances.

    """
    def __init__(self,
                 class_path: str,
                 language: Union[LanguageEnum, str],
                 class_name: str = "",
                 prompt_file: str = "",
                 prompt_dict: dict = None,
                 **kwargs):
        """
        :param class_path (str): The path specifying the location of the class.
        :param language (LanguageEnum or str): The natural language used for processing.
        :param class_name (str, optional): The name of the class. Defaults to "".
        :param prompt_file (str, optional): The file containing prompt templates. Defaults to "".
        :param prompt_dict (dict, optional): A dictionary containing prompt templates. Defaults to None.
        :param **kwargs: Additional keyword arguments.
        """
        super().__init__(class_path, language, class_name, prompt_file, prompt_dict, **kwargs)

    @staticmethod
    def get_var_name_in_str(template: str):
        """
        Extract variable names from a string template.

        :param template (str): A string template containing variables in curly braces.

        :return A list of variable names.
        """
        variables = re.findall(r'\{(\w+)\}', template)
        return variables

    @staticmethod
    def merge_dict(dict_list: List[Dict]):
        """
        Merge all dictionaries in a list of dictionaries.

        :param dict_list (List[Dict]): A list of dictionaries to be merged.

        :return A single merged dictionary.
        """
        merged_dict = {}
        for item in dict_list:
            if isinstance(item, dict):
                for k, v in item.items():
                    merged_dict[k] = v
        return merged_dict

    def template_filed_check(self, template: str, fill_dict: dict):
        """
        Check if all variables in a template are present in the fill dictionary.

        :param template (str): The template string.
        :param fill_dict (dict): The dictionary used to fill the template.

        :return A boolean value indicating whether all variables in the template are present in `fill_dict`.
        """
        template_var_list = self.get_var_name_in_str(template)
        pass_check = True
        for var in template_var_list:
            if var not in fill_dict.keys():
                logger.error(f"{var} is not in fill_dict: {fill_dict}")
                pass_check = False
                return pass_check
        return pass_check

    def get_prompt_demo_str(self, retrieved_examples: list) -> str:
        """
        Generate a demonstration prompt string based on retrieved examples.

        :param retrieved_examples (list): A list of retrieved examples.

        :return A prompt demonstration string generated from the examples and template.
        """
        example_template = self.prompt_dict.get('example_template')
        print(retrieved_examples[0].keys())
        assert self.template_filed_check(example_template, retrieved_examples[0])
        return '\n'.join(example_template.format_map(example) for example in retrieved_examples)

    def organize_icl_prompt(self, selection_examples, cur_query, configs: dict = None) -> str:
        """
        Organize an ICL prompt string.

        :param selection_examples (list): A list of selected examples.
        :param cur_query (dict): The current query dictionary.
        :param configs (dict, optional): An optional dictionary containing additional configuration information. Defaults to None.

        :return An organized ICL prompt string.
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