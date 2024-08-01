import json
from meta_icl.core.utils import (
    # get_embedding,
    sav_json,
    load_json_file)
from meta_icl.core.utils import load_jsonl, get_current_date, load_file, organize_text_4_embedding

import re, os, time
import numpy as np
import dashscope
from http import HTTPStatus

from typing import Generator, List
from loguru import logger
import requests

# KEY = ""
# KEY = "***REMOVED***"
# inl_key = ""
# dashscope.api_key = inl_key
DASHSCOPE_MAX_BATCH_SIZE = 25
def batched(inputs: List,
            batch_size: int = DASHSCOPE_MAX_BATCH_SIZE) -> Generator[List, None, None]:
    for i in range(0, len(inputs), batch_size):
        yield inputs[i:i + batch_size]

def get_embedding(inputs: list,
                  embedding_model='text_embedding_v2'):
    # inl_key = ""
    # dashscope.api_key = inl_key

    # dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
    # url = 'http://pre-dashscope-intl.aliyuncs.com/api/v1'
    #
    # headers = {'X-Request-Id': "request_id",
    #            'Content-Type': 'application/json',
    #            # 'X-DashScope-Service': 'SYS_SFT_ENHANCE_SYS_APP',
    #            'X-DashScope-Uid': "tests"}
    #
    # data = {'model': embedding_model, 'input': inputs, }
    # resp = requests.post(url=url, headers=headers, data=json.dumps(data))
    # print(resp.status_code)
    # # return_dict = json.loads(resp.keys())
    # print(resp)

    if embedding_model == 'text_embedding_v1' or embedding_model == 'text_embedding_v2':
        print(dashscope.api_key)
        print(f"dashscope.base_http_api_url: {dashscope.base_http_api_url}")
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
    return result

def get_intention_list(example_pth, intention_key, intention_config=None):
    if intention_config is not None:
        return load_json_file(intention_config)["intention_class"]
    else:
        examples = load_json_file(example_pth)
        intention_list = []
        for item in examples:
            if item[intention_key] not in intention_list:
                intention_list.append(item[intention_key])
        return intention_list


def search_str_operator(search_key_list, example):
    pass


def extract_user_prompt(text):
    match = re.search(r"\n\n输入的prompt：(.*?)\n优化后的prompt：", text, re.DOTALL)

    if match:
        extracted_text = match.group(1)
        print("Extracted text between the two prompts:")
        print(extracted_text)
        return extracted_text
    else:
        return None


def get_example_embeddings(example_list, search_key, embedding_model="text_embedding_v1"):
    """

    :param example_list: list of dict, each is an example.
    :param search_key: str, the key to search embedding
    :param embedding_model: the model to get the embedding
    :return: List of vector
    """
    # text_list = [example[search_key] for example in example_list]
    text_list = organize_text_4_embedding(example_list=example_list, search_key=search_key)

    example_embeddings = get_embedding(text_list, embedding_model=embedding_model)
    # print(example_embeddings.keys())
    # query_embedding_list = [item['embedding'] for item in example_embeddings.output['embeddings']]

    if embedding_model == "text_embedding_v1" or embedding_model == "text_embedding_v2":
        query_embedding_list = [item['embedding'] for item in example_embeddings.output['embeddings']]
    else:
        query_embedding_list = [item['embedding'] for item in example_embeddings['output']['embeddings']]
    return query_embedding_list


def sav_embeddings_to_new_file(embedding_array, sav_dir, embedding_model, prefix='', ):
    """

    :param query_embedding_list: list of embeddings
    :param sav_dir:
    :param prefix:
    :return:
    """
    cur_time = get_current_date()
    embedding_sav_pth = os.path.join(sav_dir, f'{prefix}_embModel:{embedding_model}_examples_ver_{cur_time}.npy')
    np.save(embedding_sav_pth, embedding_array)


def update_example(example_pth, search_key,
                   sav_dir, icl_config_pth,
                   sav_type='npy',
                   prefix='',
                   embedding_model="text_embedding_v1",
                   eval_key_list=None
                   ):
    from meta_icl.core.utils import check_dir
    check_dir(sav_dir)
    cur_time = get_current_date()
    from meta_icl.core.utils import convert_json_2_xlx
    if example_pth.split('.')[-1] == "xlsx":
        json_path = os.path.join(sav_dir, f"{prefix}_icl_examples_ver_{cur_time}.json")
        from meta_icl.core.utils import convert_xlsx_2_json
        example_list = convert_xlsx_2_json(json_path, example_pth, eval_key_list=eval_key_list)
        example_list_pth = json_path

        excel_file_path = os.path.join(sav_dir, f"{prefix}_icl_examples_ver_{cur_time}.xlsx")
        convert_json_2_xlx(json_path, excel_file_path)


    elif example_pth.split('.')[-1] == "json":
        example_list = load_json_file(example_pth)

        excel_file_path = os.path.join(sav_dir, f"{prefix}_icl_examples_ver_{cur_time}.xlsx")
        convert_json_2_xlx(example_pth, excel_file_path)
        json_path = os.path.join(sav_dir, f"{prefix}_icl_examples_ver_{cur_time}.json")
        sav_json(example_list, json_path)
        example_list_pth = json_path

    else:
        raise ValueError("currently support .xlsx and .json, other types need to add load data function!")
    print(f"xlsx path: {excel_file_path}\njson path: {json_path}")
    embedding_sav_pth = build_example_stock(example_list, search_key, sav_dir, sav_type='npy', prefix=prefix,
                                            embedding_model=embedding_model,
                                            cur_time=cur_time)
    update_icl_configs(config_pth=icl_config_pth,
                       embedding_pth=embedding_sav_pth,
                       embedding_model=embedding_model,
                       examples_list_pth=example_list_pth,
                       search_key=search_key)
    return None


def build_example_stock(example_list,
                        search_key,
                        sav_dir,
                        sav_type='npy',
                        prefix='',
                        embedding_model="text_embedding_v1",
                        cur_time=None) -> str:
    """
    get the embeddings of the content in the "search_key" of the example in the example_list,
    and sav it to f'{prefix}_examples_ver_{cur_time}.{sav_type}'

    :param example_list: list
    :param search_key: the key for get embedding & sample search
    :param sav_dir: the directory to save the embeddings
    :param sav_type: embedding save file type, default 'npy'
    :param prefix: the prefix of the embedding save file name
    :return: embedding_sav_pth: the pth saves the embedding.
    """
    # text_list = [example[search_key] for example in example_list]
    # example_embeddings = get_embedding(text_list)
    #
    # query_embedding_list = [item['embedding'] for item in example_embeddings.output['embeddings']]
    query_embedding_list = get_example_embeddings(example_list, search_key, embedding_model)
    if cur_time is not None:
        pass
    else:
        cur_time = get_current_date()

    embedding_array = np.vstack(query_embedding_list)

    embedding_sav_pth = os.path.join(sav_dir, "model_opt_v2_emb_model_text_embedding_v2_examples_ver_2024_05_15_22_41_44.npy")
                                     # f'{prefix}_emb_model:<{embedding_model}>_search_key:{search_key}_examples_ver_{cur_time}.npy')

    np.save(embedding_sav_pth, embedding_array)
    return embedding_sav_pth


def add_example_stock(exist_example_embedding_pth,
                      examples_to_add,
                      search_key,
                      replace=False,
                      sav_dir=None) -> None:
    """

    :param exist_examples_pth:
    :param examples_to_add:
    :param search_key:
    :param replace:
    :param sav_dir:
    :return:
    """

    # TODO
    if replace:
        pass
    else:
        assert sav_dir is not None
        # if not replace the existing embedding file, the sav dir must be provided.

    query_embedding_list = get_example_embeddings(examples_to_add, search_key)
    exist_embeddings = load_file(exist_example_embedding_pth)
    tmp = exist_embeddings.tolist()
    tmp.extend(query_embedding_list)
    updated_embeddings = np.vstack(tmp)
    if replace:
        np.save(exist_example_embedding_pth, updated_embeddings)
    else:
        sav_embeddings_to_new_file(updated_embeddings, sav_dir=sav_dir)


def update_intention_class(examples_list_pth, intention_pth, intention_key="intention_class"):
    intention_list = get_intention_list(example_pth=examples_list_pth,
                                        intention_key=intention_key)
    intention_config = load_json_file(intention_pth)
    print(f"previous: {intention_config['intention_class']}\nupdated to {intention_list}")
    intention_config['intention_class'] = intention_list
    sav_json(intention_config, intention_pth)
    return intention_list


def update_icl_configs(config_pth, embedding_pth, embedding_model, examples_list_pth, search_key):
    """

    :param config_pth: prefinded config pth, with configs like: {
  "icl_configs": {
    "base_model": "Qwen_70B",
    "embedding_pth": "data/intention_analysis_examples_emb_model:<text_embedding_v1>_examples_ver_2024-05-23 09:54:08.npy",
    "examples_pth":"data/user_query_intention_examples.json",
    "topk": 3,
    "embedding_model": "text_embedding_v1"
    "search_key": "user_query"
    }
}
    :param embedding_pth: the embedding pth
    :return:
    """

    configs = load_json_file(config_pth)
    print("load the config from: {}\nprevious embedding_pth: {}\nupdated to: {}".format(
        config_pth,
        configs["icl_configs"]["embedding_pth"],
        embedding_pth))
    configs["icl_configs"]["embedding_pth"] = embedding_pth
    if configs["icl_configs"]["embedding_model"] == embedding_model:
        pass
    else:
        print("previous embedding_model: {}\nupdated to: {}".format(

            configs["icl_configs"]["embedding_model"],
            embedding_model))
        configs["icl_configs"]["embedding_model"] = embedding_model

    if configs["icl_configs"]["examples_pth"] == examples_list_pth:
        pass
    else:
        print("previous examples_list_pth: {}\nupdated to: {}".format(
            configs["icl_configs"]["examples_pth"],
            examples_list_pth))
        configs["icl_configs"]["examples_pth"] = examples_list_pth

    configs["icl_configs"]["embedding_key"] = search_key

    sav_json(configs, config_pth)


'''
save the examples embeddings as json file: {"embeddings": List of vector, "examples": List of dict, "search_key": str}
'''
if __name__ == '__main__':
    # build_embedding_config_pth = "conf/stock_embedding_build_configs/app_emb_configs_str.json"
    # # build_embedding_config_pth = "conf/stock_embedding_build_configs/app_emb_configs_multimodal.json"
    # emb_build_configs = load_json_file(build_embedding_config_pth)
    # embedding_model = emb_build_configs["embedding_model"]
    # examples_list_pth = emb_build_configs["examples_list_pth"]
    # icl_config_pth = emb_build_configs["icl_config_pth"]
    # search_key = emb_build_configs["search_key"]
    # sav_dir = emb_build_configs["sav_dir"]
    # eval_key_list = emb_build_configs["eval_key_list"]
    # prefix = emb_build_configs["prefix"]
    #
    # update_example(example_pth=examples_list_pth, search_key=search_key, prefix=prefix,
    #                sav_dir=sav_dir, icl_config_pth=icl_config_pth,
    #                embedding_model=embedding_model,
    #                eval_key_list=eval_key_list
    #                )
    # input_list = ["你好呀"]
    # print(get_example_embeddings(input_list, embedding_model="text_embedding_v2"))

    example_list = load_json_file(json_file_path="log/prompt_opt_examples_ver_2.json")
    search_key = "query"
    sav_dir = './'

    build_example_stock(example_list,
                        search_key,
                        sav_dir,
                        sav_type='npy',
                        prefix='',
                        embedding_model="text_embedding_v2",
                        cur_time=None)
