from typing import List, Dict
import random, json

from datetime import datetime

from meta_icl.core.utils.sys_prompt_utils import get_embedding, message_formatting, call_llm_with_message, sav_json
import time
import logging
import re
from functools import wraps
from loguru import logger



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


def extract_from_markdown_json(text):
    """
    extract the json from markdown text to a list of dictionaries.
    :param text:
    :return results_list: list of dict
    """
    logger.info("[extract_from_markdown_json] input_text: \n{}\n\n".format(text))
    # pattern = r'```json\n(.*?)```'
    # match = re.search(pattern, text, re.DOTALL)

    # matches = re.findall(r'```json\n\s+(.+?)\s+```', text, re.DOTALL)
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
    from meta_icl.core.utils.sys_prompt_utils import sav_json
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
        from meta_icl.core.utils.sys_prompt_utils import load_json_file
        return load_json_file(file_pth)
    elif file_type == 'josnl':
        return load_jsonl(file_pth)
    elif file_type == 'csv':
        from meta_icl.core.utils.sys_prompt_utils import load_csv
        return load_csv(file_pth)
    else:
        ValueError(f'cannot support file type: {file_type}!')


def organize_text_4_embedding(example_list, search_key):
    """

    :param example_list: list of dict
    :param search_key: str or list of str
    :return: list of str for the embedding

    Notice: if search_key is srt or len(search_key) ==1, then directly use the value of that search_key.
    If len(search_key) > 1: reformatted as:  ", ".join(f"{search_key_name}: {example[search_key_name]}"
                            for search_key_name in search_key)
    """

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
                    ", ".join(f"{search_key_name}: {example[search_key_name]}"
                              for search_key_name in search_key)
                    for example in example_list]
        else:
            raise ValueError("search_key must be str or list type")
    else:
        text_list = example_list

    return text_list


def get_single_embedding(query, embedding_model, search_key=None):
    """

    :param query: str
    :return: embedding vector
    """

    query = organize_text_4_embedding(example_list=query, search_key=search_key)
    print(f"rewrite search query for embedding as: {query}")
    try:
        return get_embedding(query, embedding_model=embedding_model).output['embeddings'][0]['embedding']
    except:
        return get_embedding(query, embedding_model=embedding_model)["output"]['embeddings'][0]['embedding']


def combine_session(csv_pth, json_sav_dir, group_by_filed, selection_filed=None, prefix="", mapping_filed=None):
    import pandas as pd
    import os

    data = pd.read_csv(csv_pth)
    print(data.keys())
    sav_dir = json_sav_dir

    grouped = data.groupby(group_by_filed)
    sessions = []
    for session_id, group in grouped:
        tmp = group.to_dict(orient='records')
        conversations = []
        # print(tmp)
        print(len(tmp))

        if selection_filed is not None:
            if mapping_filed is not None:
                pass
            else:
                mapping_filed = selection_filed

            for item in tmp:
                empty_dict = {}
                for key_id in range(len(selection_filed)):
                    empty_dict[mapping_filed[key_id]] = item[selection_filed[key_id]]
                conversations.append(empty_dict)
        else:
            conversations = tmp

        session_dict = {
            'session_id': session_id,
            'conversations': conversations  # List of rows for this session
        }

        sessions.append(session_dict)
    sav_json(sessions, os.path.join(sav_dir, f"{prefix}_ver_{get_current_date()}.json"))
    return sessions


# Example usage
if __name__ == "__main__":
    import numpy as np


    # Define a toy problem.
    def example_expand_fn(state):
        # Produce some new states from the current state.
        # In a real application, this would involve generating possible next steps.
        import copy
        expand = []
        for _ in range(3):
            state_copy = copy.deepcopy(state)
            state_copy.append(np.random.choice(5))
            expand.append(state_copy)
        return expand


    def example_score_fn(state):
        # Score the state. In a real application, this score could be based on
        # model predictions, heuristics, or other criteria.
        # This toy example prefers states with more 'a's.
        return np.sum(state)


    initial_state = [1]
    max_steps = 5
    beam_width = 2
    best_state = beam_search(initial_state, max_steps, beam_width, example_expand_fn, example_score_fn)

    print(f"Best state found: {best_state}")
