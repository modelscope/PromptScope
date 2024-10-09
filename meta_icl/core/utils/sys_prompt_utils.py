import copy
import random
import time
from http import HTTPStatus
from typing import Generator, List
from typing import Union
import re
import json
import requests

import dashscope
import numpy as np
from loguru import logger
from scipy.spatial.distance import cdist

from meta_icl.core.models.generation_model import GenerationModel

KEY = ""

# # KEY = "***REMOVED***"
dashscope.api_key = KEY

DASHSCOPE_MAX_BATCH_SIZE = 25
DefaultModelConfig = {
    'model': 'qwen-max',
    'seed': 1234,
    'result_format': 'message',
    'temperature': 0.85
}


def convert_model_name_to_model_config(model_name: Union[str, dict] = None,
                                       add_random=False,
                                       model_config=None, **kwargs) -> dict:
    if model_name is not None:
        if model_name.lower() == 'qwen_200b':
            model_name = 'qwen_max'
        elif model_name.lower() == 'qwen_14b':
            model_name = 'qwen-turbo'
        elif model_name.lower() == 'qwen_70b':
            model_name = 'qwen-plus'

        model_config = copy.deepcopy(DefaultModelConfig)
        model_config['model'] = model_name

        if add_random:
            model_config['seed'] = np.random.randint(1, 10000)
            model_config['temperature'] = 1.2

        else:
            model_config['model'] = model_name
            for key, value in kwargs.items():
                model_config[key] = value
        return model_config
    else:
        assert model_config is not None
        # model_config:
        # module_name: 'aio_generation'
        # model_name: qwen - max - allinone
        # clazz: 'models.llama_index_generation_model'
        # max_tokens: 2000
        # seed: 1234
        # temperature: 1
        model_config_add_random = copy.deepcopy(model_config)
        if add_random:
            model_config_add_random['seed'] = np.random.randint(1, 10000)
            model_config['temperature'] = 1.2
        return model_config_add_random


def find_top_k_embeddings(query_embedding, list_embeddings, k):
    '''

    :param query_embedding:
    :param list_embeddings:
    :param k:
    :return: List of List: each element is (index, embedding, score)
    '''
    # Compute cosine similarity between the query and the list of embeddings.
    # cdist returns the distance, so we subtract from 1 to get similarity.
    # Cosine distance is defined as 1.0 minus the cosine similarity.
    similarities = 1 - cdist([query_embedding], list_embeddings, 'cosine').flatten()

    # Get the top k indices sorted by similarity (in descending order).
    top_k_indices = np.argsort(similarities)[::-1][:k]

    # Get the top k embeddings and their corresponding similarity scores.
    top_k_embeddings = [list_embeddings[i] for i in top_k_indices]
    top_k_scores = [similarities[i] for i in top_k_indices]

    # Return the top k embeddings, similarity scores, and their indices.
    return [(index, embedding, score) for index, embedding, score in zip(top_k_indices, top_k_embeddings, top_k_scores)]


def sample_elements_and_ids(lst, k):
    indexed_elements = list(enumerate(lst))

    # random select k items
    selected_samples = random.sample(indexed_elements, k)

    # get the item and its index
    selected_indices = [index for index, element in selected_samples]
    selected_elements = [element for index, element in selected_samples]

    return selected_elements, selected_indices


def check_dir(dir_pth):
    import os
    if os.path.exists(dir_pth):
        pass
    else:
        os.mkdir(dir_pth)


def load_csv(pth):
    import csv

    # Initialize an empty dictionary to hold our column data
    csv_columns = {}

    # Open the CSV file
    with open(pth, mode='r') as csvfile:
        # Create a csv.DictReader object
        reader = csv.DictReader(csvfile)

        # Initialize our dictionary with empty lists for each column
        for column in reader.fieldnames:
            csv_columns[column] = []

        # Iterate through the rows in the CSV file
        for row in reader:
            # For each column, append the data to the list in our dictionary
            for column in reader.fieldnames:
                csv_columns[column].append(row[column])

    # At this point, csv_columns will contain the data from the CSV file,
    # with each column as a list of values, keyed by the column name
    return csv_columns


def call_llm_with_message(messages, model: Union[str, GenerationModel], model_config=None, is_stream=False, **kwargs):
    """
    :param messages: the messages to call the model;
    :param model: the model to call; Default: False;
    support: "gpt4", "Qwen_200B", "Qwen_14B", "Qwen_70B", "qwen2*", dashscope base models
    :param model_config: the model config, example: {
    'model': 'qwen-max',
    'seed': 1234,
    'result_format': 'message',
    'temperature': 0.85
}
Attention: If both model and model_config are given, the model_config will be used.
    :param is_stream: whether the stream output is required. Default: False
    :param kwargs: other keyword arguments
    """
    # print("\n***** messages *****\n{}\n".format(messages))
    if isinstance(model, GenerationModel):
        res = model.call(messages=messages, stream=is_stream, **kwargs)
        logger.info(res)
        # return res.output.text
        return res.message.content

    if is_stream and model.lower() != 'qwen_200b':
        raise ValueError("expect Qwen model, other model's stream output is not supported")
    logger.info(model_config)
    if model.lower() == 'gpt4':
        return call_gpt_with_message(messages, **kwargs)
    elif model.lower() == 'qwen_200b' or model.lower() == 'qwen-max':
        if model_config is not None:
            pass
        else:
            model_config = {
                'model': 'qwen-max',
                'seed': 1234,
                'result_format': 'message'
            }
        if is_stream:
            pass
            # return call_qwen_with_stream(messages, model_config=model_config, **kwargs)
        else:
            return call_qwen_with_message_with_retry(messages, model_config=model_config, **kwargs)
    elif model.lower() == 'qwen_14b' or model.lower() == 'qwen-turbo':
        if model_config is not None:
            pass
        else:
            model_config = {
                'model': 'qwen-turbo',
                'seed': 1234,
                'result_format': 'message'
            }

        return call_qwen_with_message_with_retry(messages,
                                                 model_config=model_config, **kwargs)
    elif model.lower() == 'qwen_70b' or model.lower() == 'qwen-plus':
        if model_config is not None:
            pass
        else:
            model_config = {
                'model': 'qwen-plus',
                'seed': 1234,
                'result_format': 'message',
                'temperature': 1.0
            }

        return call_qwen_with_message_with_retry(messages,
                                                 model_config=model_config, **kwargs)
    elif model.lower() == "qwen2-72b-instruct":
        if model_config is not None:
            pass
        else:
            model_config = {
                'model': 'qwen2-72b-instruct',
                'seed': 1234,
                'result_format': 'message',
                'temperature': 1.0
            }

        return call_qwen_with_message_with_retry(messages,
                                                 model_config=model_config, **kwargs)
    elif model.lower().split('-')[0] == "qwen2":
        if model_config is not None:
            pass
        else:
            model_config = {
                'model': model,
                'seed': 1234,
                'result_format': 'message',
                'temperature': 0.85
            }

        return call_qwen_with_message_with_retry(messages,
                                                 model_config=model_config, **kwargs)
    else:
        dashscope.base_http_api_url = 'https://poc-dashscope.aliyuncs.com/api/v1'
        dashscope.base_websocket_api_url = 'https://poc-dashscope.aliyuncs.com/api-ws/v1/inference'
        try:
            model_config = {
                'model': model,
                'seed': 1234,
                'result_format': 'message'
            }
            return call_qwen_with_message_with_retry(messages, model_config=model_config, **kwargs)

        except Exception:
            raise ValueError('model: {} is not supported!'.format(model))


def sav_json(data, json_file_path):
    with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False, indent=4)


def load_json_file(json_file_path):
    """
    Load data from a JSON file.

    Parameters:
    - json_file_path: str, the path to the JSON file.

    Returns:
    - data: The data loaded from the JSON file.
    """
    with open(json_file_path, 'r', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
    return data


def load_txt(instruction_file: str) -> str:
    """
    load .txt file
    Arguments:
        instruction_file: str type, which is the .txt file pth

    Returns:
        instruction: str type, which is the str in the instruction_file

    """
    with open(instruction_file, "r", encoding="utf-8") as f:
        instruction = f.read()
    return instruction


def sav_csv(data, pth):
    import csv

    # Specify the CSV file name
    file_path = pth

    # Open the file for writing
    with open(file_path, mode='w', encoding='utf-8') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)

        # Write the header (dictionary keys)
        writer.writerow(data.keys())

        # Write the rows (dictionary values)
        writer.writerows(zip(*data.values()))


def fill_in_variables(variables, template_text):
    # 将模板中的${variable}替换为{variable}风格
    for key in variables:
        # 使用replace方法替换模板中的占位符
        template_text = template_text.replace('${' + key + '}', variables[key])
    # for key in variables:
    #     template_text = template_text.replace('${' + key + '}', '{' + key + '}')

    # 使用变量填充模板文本
    # formatted_text = template_text.format(**variables)

    return template_text


def call_qwen_with_message_with_retry(messages,
                                      model_config=DefaultModelConfig, **kwargs):
    # cnt = 0
    # error_message = ""
    # print('*' * 10 + '\nworking on id: {}, \ninput: {}\n'.format(id, prompt_temp[id]))
    try:
        _, res = call_qwen_with_messages(messages,
                                         model_config=model_config,
                                         **kwargs)
        logger.info(res)
        return res['output']['choices'][0]['message']['content']
    except Exception as e:
        logger.error("\n\nmessages: {}".format(e))


def call_qwen_with_messages(messages, model_config=DefaultModelConfig, **kwargs):
    X_DashScope_EUID = kwargs.get('X_DashScope_EUID')

    if "temperature" in model_config.keys():
        temperature = model_config['temperature']
    else:
        temperature = 0.85
    response = dashscope.Generation.call(
        model_config['model'],
        # dashscope.Generation.Models.qwen_max,
        messages=messages,
        # set the random seed, optional, default to 1234 if not set
        seed=model_config['seed'],
        result_format='message',  # set the result to be "message" format.
        temperature=temperature,
        headers={"X-DashScope-Euid": X_DashScope_EUID},

    )
    # logger.query_info(request_id=request_id,
    #                   message='X_DashScope_EUID: {}, dash return: {}'.format(X_DashScope_EUID, response))
    if response.status_code == HTTPStatus.OK:
        logger.info(response)
        return True, response
    else:
        return (False,
                'Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                    response.request_id, response.status_code,
                    response.code, response.message
                ))


def call_qwen_with_prompt(prompt, model_config=DefaultModelConfig):
    response = dashscope.Generation.call(
        model=model_config['model'],
        seed=model_config['seed'],
        prompt=prompt
    )
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    if response.status_code == HTTPStatus.OK:
        # print(response)
        return True, response
    else:
        return (False,
                'Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                    response.request_id, response.status_code,
                    response.code, response.message
                ))


def call_with_messages():
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '如何做西红柿炒鸡蛋？'}]
    response = dashscope.Generation.call(
        'qwen-max',
        # dashscope.Generation.Models.qwen_max,
        messages=messages,
        # set the random seed, optional, default to 1234 if not set
        # seed=random.randint(1, 10000),
        result_format='message',  # set the result to be "message" format.
    )
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print(
            'Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))


def call_gpt_with_message(messages, model_config={'model': 'gpt-4'}):
    def get_response(messages, model_config):
        url = "https://api.mit-spider.alibaba-inc.com/chatgpt/api/ask"
        headers = {
            "Content-Type": "application/json",
            "Authorization": ""
        }
        data = {
            "model": model_config['model'],
            "messages": messages,
            "n": 1,
            "temperature": 0.4
        }
        response = requests.post(url, json=data, headers=headers)
        return response.json()

    cnt = 0
    while cnt < 2:
        completion = None
        try:
            res = get_response(messages, model_config)
            return res["data"]["response"]["choices"][0]["message"]["content"]
        except Exception as e:
            print(completion)
            print('Error {}, sleep 3s and retry'.format(e))
            cnt += 1
            time.sleep(3)
    return None


def get_topk_indices(lst, k):
    # Create a list of (index, element) pairs
    indexed_list = list(enumerate(lst))
    # Sort the list by elements in descending order
    sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
    # Get the top k elements
    topk_elements = sorted_indexed_list[:k]
    # Extract the indices
    topk_indices = [index for index, element in topk_elements]
    return topk_indices


def message_formatting(system_prompt=None, query='', history=None):
    if history is not None:
        if history[0]['role'] == 'system':
            history.append({'role': 'user', 'content': query})
        else:
            messages = [
                {
                    'role': 'system', 'content': system_prompt
                }
            ]
            messages.extend(history)
            messages.append(
                {'role': 'user', 'content': query}
            )
            return messages
    else:
        if system_prompt is not None:
            return [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': query}
            ]
        else:
            return [
                {'role': 'system', 'content': "You are a helpful assistant."},
                {'role': 'user', 'content': query}
            ]


def extract_feedbacks(feedback):
    # 使用正则表达式匹配<START>和<END>之间的内容
    matches = re.findall(r'<START>(.*?)<END>', feedback, re.DOTALL)

    # 将匹配到的内容放入列表中
    suggestions = [match.strip() for match in matches]
    return suggestions


def batched(inputs: List,
            batch_size: int = DASHSCOPE_MAX_BATCH_SIZE) -> Generator[List, None, None]:
    for i in range(0, len(inputs), batch_size):
        yield inputs[i:i + batch_size]


def embed_with_list_of_str(inputs: List, embedding_model='ds_text_embedding_v1'):
    if embedding_model == 'text_embedding_v1' or embedding_model == 'text_embedding_v2':
        if embedding_model == 'text_embedding_v1':
            embedding_model = dashscope.TextEmbedding.Models.text_embedding_v1
        else:
            embedding_model = dashscope.TextEmbedding.Models.text_embedding_v2
        result = None  # merge the results.
        batch_counter = 0
        for batch in batched(inputs):
            resp = dashscope.TextEmbedding.call(
                model=embedding_model,
                input=batch)
            if resp.status_code == HTTPStatus.OK:
                if result is None:
                    result = resp
                else:
                    for emb in resp.output['embeddings']:
                        emb['text_index'] += batch_counter
                        result.output['embeddings'].append(emb)
                    result.usage['total_tokens'] += resp.usage['total_tokens']
            else:
                print(resp)
            batch_counter += len(batch)

    elif embedding_model == 'pre-bge_small_zh_v1_5-1958':
        print(f"embedding_model: {embedding_model}")
        print(inputs)

        def test_emb_request(model, text_input):
            url = 'https://poc-dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding'
            headers = {
                'Authorization': '',
                'Content-Type': 'application/json',
            }
            data = {
                'model': model,
                'input': {
                    'texts': text_input
                }
            }
            response = requests.post(url, data=json.dumps(data), headers=headers)
            return response.json()

        batch_counter = 0
        result = None
        for batch in batched(inputs, batch_size=15):
            print(f"batch_counter: {batch_counter}")
            resp = test_emb_request(embedding_model, batch)
            print(resp)
            print(resp.keys())
            if result is None:
                result = resp
            else:
                for emb in resp["output"]['embeddings']:
                    emb['text_index'] += batch_counter
                    result["output"]['embeddings'].append(emb)
            batch_counter += len(batch)
            print(f"batch_counter: {batch_counter}")
    else:

        print(f"not supported emb_model: {embedding_model}")
        result = []
    return result


def get_embedding(input_list: list, embedding_model="text_embedding_v1"):
    return embed_with_list_of_str(input_list, embedding_model=embedding_model)


def text_rerank(query, documents, top_n=None):
    """
    :param query: str
    :param documents: list of str
    :param top_n: int, the number of relevance text returned.
    """
    if top_n is not None:
        pass
    else:
        top_n = len(documents)
    resp = dashscope.TextReRank.call(
        model=dashscope.TextReRank.Models.gte_rerank,
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=True
    )
    if resp.status_code == HTTPStatus.OK:
        print(resp)
        return resp
    else:
        print(resp)
        raise Exception(resp)


def call_llama_with_messages():
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '介绍下故宫？'}]
    response = dashscope.Generation.call(
        # model='llama2-7b-chat-v2',
        model="llama3-70b-instruct",
        messages=messages,
        result_format='message',  # set the result to be "message" format.
    )
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
