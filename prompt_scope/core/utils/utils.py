import json
import random
import re
import sys
import time
from datetime import datetime
from functools import wraps
from typing import List, Dict

import yaml
from easydict import EasyDict as edict
from loguru import logger

from prompt_scope.core.utils.sys_prompt_utils import (get_embedding, message_formatting, call_llm_with_message)


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Function {func.__name__!r} execute with {duration:.4f} s")
        return result

    return wrapper


def add_duration(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Function {func.__name__!r} execute with {duration:.4f} s")
        return result, duration

    return wrapper


def extract_from_markdown_json(text):
    """
    extract the json from markdown text to a list of dictionaries.
    :param text:
    :return results_list: list of dict
    """
    logger.info("[extract_from_markdown_json] input_text: \n{}\n\n".format(text))

    matches = re.findall(r"```json\n(.*?)\n```", text, re.DOTALL)
    results_list = []
    logger.info("matches: \n{}\n".format(matches))
    for match in matches:
        try:
            # data_dict = eval(match)
            data_dict = match.replace("\n", "\\n")
            logger.info("try1: \n{}\n".format(data_dict))
            data_dict = json.loads(data_dict)
            results_list.append(data_dict)

        except json.JSONDecodeError as e:
            logger.info("try1: cannot decode JSON string: ", e)
            try:
                data_dict = """{}""".format(match)
                data_dict = json.loads(f'[{data_dict}]')
                results_list.extend(data_dict)
            except json.JSONDecodeError as e:
                logger.info("try2: cannot decode JSON string: ", e)
                try:
                    data_dict = match.replace("\n", "\\n")
                    logger.info("try3: \n{}\n".format(data_dict))
                    data_dict = json.loads(f'[{data_dict}]')
                    results_list.extend(data_dict)
                except json.JSONDecodeError as e:
                    logger.info("try4: cannot decode JSON string: ", e)
                    try:
                        messages = message_formatting(system_prompt=None,
                                                      query=f"convert the following text to the json string that can by executed by json.loads() in python. Please directly output the answer. text:\n{match}")
                        results = call_llm_with_message(messages=messages, model="Qwen_200B")
                        logger.info("refine by qwen_200B: {}".format(results))
                        results = results.replace("```json", "```")
                        data_dict = json.loads(results)
                        results_list.append(data_dict)
                    except json.JSONDecodeError as e:
                        logger.info("cannot decode JSON string: ", e)
                        # raise ValueError(f"cannot decode JSON string: {e}")
    return results_list


def get_current_date():
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return current_time


def sav_dict_2_xlsx(data, pth):
    import pandas as pd
    df = pd.DataFrame(data)

    # Save to Excel file
    df.to_excel(pth, index=False)


def sav_csv(data: dict, pth):
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv(pth, index=False)


def convert_json_2_xlx(json_file_path, excel_file_path):
    # Step 1: Load the JSON file into a pandas DataFrame
    import pandas as pd
    df = pd.read_json(json_file_path)

    # Step 2: Convert the DataFrame to an Excel file
    df.to_excel(excel_file_path,
                index=False)


def convert_xlsx_2_json(json_file_path, excel_file_path, eval_key_list=()):
    import pandas as pd

    # Read the xlsx file
    df = pd.read_excel(excel_file_path)
    if isinstance(eval_key_list, str):
        eval_key_list = [eval_key_list]

    # Convert the DataFrame to a list of dictionaries
    data_dicts = df.to_dict(orient='records')
    for idx in range(len(data_dicts)):
        print(data_dicts[idx])
        for key in data_dicts[idx].keys():
            if key in eval_key_list:
                print(data_dicts[idx][key])
                data_dicts[idx][key] = eval(data_dicts[idx][key])

    print(data_dicts)

    # Print the JSON data
    from prompt_scope.core.utils.sys_prompt_utils import sav_json
    sav_json(data=data_dicts, json_file_path=json_file_path)
    return data_dicts


def load_jsonl(file_path):
    tmp = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            # Parse the JSON object
            json_obj = json.loads(line)
            tmp.append(json_obj)
    return tmp


def random_selection_method(example_lists: List[Dict],
                            num: int):
    selected_list, ids = sample_elements_and_ids(example_lists, num)
    return {
        'ids': ids,
        'selected_examples': selected_list
    }


def sample_elements_and_ids(lst, k):
    indexed_elements = list(enumerate(lst))

    # random select k items
    selected_samples = random.sample(indexed_elements, k)

    # get the item and its index
    selected_indices = [index for index, element in selected_samples]
    selected_elements = [element for index, element in selected_samples]

    return selected_elements, selected_indices


def load_file(file_pth):
    file_type = file_pth.split('.')[-1]
    if file_type == 'npy':
        import numpy as np
        return np.load(file_pth)
    elif file_type == 'json':
        from prompt_scope.core.utils.sys_prompt_utils import load_json_file
        return load_json_file(file_pth)
    elif file_type == 'josnl':
        return load_jsonl(file_pth)
    elif file_type == 'csv':
        from prompt_scope.core.utils.sys_prompt_utils import load_csv
        return load_csv(file_pth)
    elif file_type == 'index':
        from faiss import read_index
        return read_index(file_pth)
    else:
        ValueError(f'cannot support file type: {file_type}!')


def organize_text_4_embedding(example_list: list, search_key):
    """

    :param example_list: list of dict
    :param search_key: str or list of str
    :return: list of str for the embedding

    Notice: if search_key is srt or len(search_key) ==1, then directly use the value of that search_key.
    If len(search_key) > 1: reformatted as:  ", ".join(f"{search_key_name}: {example[search_key_name]}"
                            for search_key_name in search_key)
    """
    logger.info(f"search_key: {search_key}")
    logger.info(f"example_list: {example_list}")

    if search_key is not None:
        # if search_key is str or len(search_key) ==1, then directly use the value of that search_key.
        if isinstance(search_key, str):
            text_list = [example[search_key] for example in example_list]
        elif isinstance(search_key, list):
            if len(search_key) == 1:
                text_list = [example[search_key[0]] for example in example_list]
            else:
                # len(search_key) > 1: concatenate the search_key_name into str.
                text_list = [
                    ", ".join(f"{search_key_name}: {str(example[search_key_name])}"
                              for search_key_name in search_key)
                    for example in example_list]
        else:
            raise ValueError("search_key must be str or list type")
    else:
        text_list = example_list
    logger.info(type(text_list[0]))

    return text_list


def get_single_embedding(query, embedding_model, search_key=None):
    """

    :param query: str
    :return: embedding vector
    """
    logger.info(f'input query: {query}')
    query = organize_text_4_embedding(example_list=query, search_key=search_key)
    logger.info(f"rewrite search query for embedding as: {query}")
    try:
        return get_embedding(query, embedding_model=embedding_model).output['embeddings'][0]['embedding']
    except Exception as e:
        logger.error(e)
        return get_embedding(query, embedding_model=embedding_model)["output"]['embeddings'][0]['embedding']


def convert_json_2_yaml(json_file_path, yaml_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
        logger.info(f"load json file: {json_file_path}")
    sav_yaml(data=data, yaml_file_path=yaml_file_path)


def load_yaml(yaml_path: str, as_edict: bool = True) -> edict:
    """
    Reads the yaml file and enrich it with more vales.
    :param yaml_path: The path to the yaml file
    :param as_edict: If True, returns an EasyDict configuration
    :return: An EasyDict configuration
    """
    print(sys.path)
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    if as_edict:
        yaml_data = edict(yaml_data)
    return yaml_data


def load_yaml_file(yaml_file_path):
    try:
        with open(yaml_file_path, 'r') as file:
            # Load the YAML content
            yaml_content = yaml.safe_load(file)
            # Print the YAML content
            print(yaml.dump(yaml_content, default_flow_style=False))
            return yaml_content
    except Exception as e:
        logger.error(f"Error loading YAML file: {e}")


def sav_yaml(data, yaml_file_path):
    yaml_text = yaml.dump(data, allow_unicode=True, sort_keys=False)
    logger.info(yaml_text)
    with open(yaml_file_path, 'w') as f:
        f.write(yaml_text)
        logger.info(f"save yaml file to: {yaml_file_path}")
