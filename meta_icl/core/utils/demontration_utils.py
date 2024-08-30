# -*- coding: utf-8 -*-
from meta_icl.core.utils.sys_prompt_utils import (call_llm_with_message, message_formatting, text_rerank,
                                                  convert_model_name_to_model_config)
import re, json, os, copy
import numpy as np

from loguru import logger
from typing import Union, Dict, List


Default_Instruction_4_Demonstration_Generation = """请根据提供的样例，给出${num_generated_examples}个类似样例，要求和现在的样例的任务类型一致。

要求：
1. 生成的语言和提供的参考样例保持一致， 即提供的参考样例是英文的，你给出的样例也应该是英文的；如果提供的参考样例是中文的，你给出的样例也应该是中文的
2. 给出的样例尽量与参考样例属于同一个任务类型，但和参考样例有较大区别，并且是不同domain的。
3. 和提供的参考样例保持一致输出格式，并且每个样例用markdown json 形式单独区分。
${other_requirements}

参考样例：
```json
${demonstration}
```

请给出${num_generated_examples}个类似样例:
"""

# other_requirements = "其他要求：\n1. \"starting_questions\" 是推荐用户问智能体的问题\n2. \"tools\"可选的范围是[\"text-to-image\", \"open-search\", \"code_interpreter\"]"


def demo_augmentation_by_llm_prompt_org(
        demonstration_text: Union[str, Dict, List[Dict]],
        demonstration_generation_instruction: str=None,
        num_generated_examples=1,
        demonstration_requirements=None
):
    """
    generate demonstration based on the reference demonstration (demonstration_text)
    :param demonstration_text:
    :param demonstration_requirements:
    :param demonstration_generation_instruction:
    :param num_generated_examples: the number of generated examples
    """
    if demonstration_generation_instruction is not None:
        logger.info("demonstration_generation_instruction is provided as: {}".format(
            demonstration_generation_instruction))
    else:
        logger.info("demonstration_generation_instruction is not provided!, use the default on: {}".format(
            Default_Instruction_4_Demonstration_Generation))
        demonstration_generation_instruction = Default_Instruction_4_Demonstration_Generation
    # extract demonstration text
    if isinstance(demonstration_text, dict):
        demonstration_text = f"```json\n{json.dumps(demonstration_text, ensure_ascii=False)}\n```"
    if isinstance(demonstration_text, List):
        if isinstance(demonstration_text[0], str):
            demonstration_text = "\n".join(demonstration_text)
        elif isinstance(demonstration_text[0], dict):
            demonstration_text = "\n".join(f"```json\n{json.dumps(item, ensure_ascii=False)}\n```"
                                           for item in demonstration_text)
            logger.info("demonstration_text: \n{}\n".format(demonstration_text))
    demonstration_generation_instruction = demonstration_generation_instruction.replace("${demonstration}",
                                                                                        demonstration_text)
    demonstration_generation_instruction = demonstration_generation_instruction.replace("${num_generated_examples}",
                                                                                        str(num_generated_examples))
    prompt = demonstration_generation_instruction.replace("${other_requirements}", demonstration_requirements)
    logger.info("prompt: \n{}\n".format(prompt))
    prompt = message_formatting(system_prompt=None, query=prompt)
    return prompt




