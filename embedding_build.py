import json
from meta_icl.utils.sys_prompt_utils import get_embedding, sav_json, load_json_file
from meta_icl.utils.utils import load_jsonl, get_current_date, load_file, organize_text_4_embedding
from meta_icl.icl.ICL import BaseRetrive
import re, os, time
import numpy as np


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
                   sav_dir, config_pth,
                   sav_type='npy',
                   prefix='',
                   embedding_model="text_embedding_v1",
                   eval_key_list=None,
                   intention_pth=None
                   ):
    from meta_icl.utils.sys_prompt_utils import check_dir
    check_dir(sav_dir)
    cur_time = get_current_date()
    from meta_icl.utils.utils import convert_json_2_xlx
    if example_pth.split('.')[-1] == "xlsx":
        json_path = os.path.join(sav_dir, f"{prefix}_icl_examples_ver_{cur_time}.json")
        from meta_icl.utils.utils import convert_xlsx_2_json
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
    update_icl_configs(config_pth=config_pth,
                       embedding_pth=embedding_sav_pth,
                       embedding_model=embedding_model,
                       examples_list_pth=example_list_pth)
    update_intention_class(examples_list_pth=example_list_pth,
                           intention_pth=intention_pth)


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

    embedding_sav_pth = os.path.join(sav_dir,
                                     f'{prefix}_emb_model:<{embedding_model}>_search_key:{search_key}_examples_ver_{cur_time}.npy')

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
  "analyzer_config": {
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
        configs["analyzer_config"]["embedding_pth"],
        embedding_pth))
    configs["analyzer_config"]["embedding_pth"] = embedding_pth
    if configs["analyzer_config"]["embedding_model"] == embedding_model:
        pass
    else:
        print("previous embedding_model: {}\nupdated to: {}".format(

            configs["analyzer_config"]["embedding_model"],
            embedding_model))
        configs["analyzer_config"]["embedding_model"] = embedding_model

    if configs["analyzer_config"]["examples_pth"] == examples_list_pth:
        pass
    else:
        print("previous examples_list_pth: {}\nupdated to: {}".format(
            configs["analyzer_config"]["examples_pth"],
            examples_list_pth))
        configs["analyzer_config"]["examples_pth"] = examples_list_pth

    configs["analyzer_config"]["embedding_key"] = search_key

    sav_json(configs, config_pth)


'''
save the examples embeddings as json file: {"embeddings": List of vector, "examples": List of dict, "search_key": str}
'''
if __name__ == '__main__':
    embedding_model = "text_embedding_v2"
    #  embedding_model = "pre-bge_small_zh_v1_5-1958"
    examples_list_pth = "data/user_query_intention_examples.json"
    # examples_list_pth = "data/test_icl_build/test_icl_build_icl_examples_ver_2024-05-28 20:46:53.xlsx"
    config_pth = "conf/base_conf.json"
    search_key = ["user_query", "chat_history"]
    sav_dir = "data/icl_examples"
    eval_key_list = "chat_history"
    prefix = "test_icl_build"
    intention_pth = "data/intention_classes.json"

    update_example(example_pth=examples_list_pth, search_key=search_key, prefix=prefix,
                   sav_dir=sav_dir, config_pth=config_pth,
                   embedding_model=embedding_model,
                   eval_key_list=eval_key_list,
                   intention_pth=intention_pth
                   )

    #
    # example_list = load_json_file(examples_list_pth)

    # sav_dir = "data/"
    # prefix = "intention_analysis_examples"
    #
    # embedding_sav_pth = build_example_stock(example_list, search_key, sav_dir, sav_type='npy', prefix=prefix,
    #                                         embedding_model=embedding_model)
    # update_icl_configs(config_pth=config_pth,
    #                    embedding_pth=embedding_sav_pth,
    #                    embedding_model=embedding_model,
    #                    examples_list_pth=examples_list_pth)

    # from meta_icl.utils.utils import convert_xlsx_2_json
    # pth = "data/user_query_intention_examples.xlsx"
    # eval_key = "chat_history"
    # convert_xlsx_2_json(json_file_path="test.json",
    #                     excel_file_path=pth,
    #                     eval_key_list=eval_key)
