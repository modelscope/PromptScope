from meta_icl.core.offline.demonstration_stock_preparation.stock_builder_4_bm25 import BaseStockBuilder
from meta_icl.core.utils.utils import (get_current_date,
                                       convert_xlsx_2_json,
                                       convert_json_2_xlx,
                                       organize_text_4_embedding)
from meta_icl.core.utils.sys_prompt_utils import load_json_file, sav_json, check_dir, get_embedding
import os
import numpy as np
from loguru import logger


def demonstration_backup(sav_dir, demonstration_pth, prefix='', eval_key_list=None):
    """
    Backs up the demonstration file and converts it into JSON and Excel formats.
    :param sav_dir: (str), The path of the directory where files will be saved.
    :param demonstration_pth: (str), The path of the original demonstration file.
    :param prefix: (str), Prefix for the backup filenames, default is an empty string.
    :param eval_key_list: List of evaluation keys, default is None.

    Returns:
    :return example_list: (list), The list of examples after conversion.
    :return example_list_pth: (str), The path of the converted JSON file.
    """
    check_dir(sav_dir)
    cur_time = get_current_date()
    if demonstration_pth.split('.')[-1] == "xlsx":
        json_path = os.path.join(sav_dir, f"{prefix}_icl_examples_ver_{cur_time}.json")
        example_list = convert_xlsx_2_json(json_path, demonstration_pth, eval_key_list=eval_key_list)
        example_list_pth = json_path
        excel_file_path = os.path.join(sav_dir, f"{prefix}_icl_examples_ver_{cur_time}.xlsx")
        convert_json_2_xlx(json_path, excel_file_path)

    elif demonstration_pth.split('.')[-1] == "json":
        example_list = load_json_file(demonstration_pth)
        excel_file_path = os.path.join(sav_dir, f"{prefix}_icl_examples_ver_{cur_time}.xlsx")
        convert_json_2_xlx(demonstration_pth, excel_file_path)
        json_path = os.path.join(sav_dir, f"{prefix}_icl_examples_ver_{cur_time}.json")
        sav_json(example_list, json_path)
        example_list_pth = json_path

    else:
        logger.error("currently support .xlsx and .json, other types need to add load data function!")
        raise ValueError("currently support .xlsx and .json, other types need to add load data function!")
    logger.info(f"demonstration backup done! The json_file_pth: {json_path}; xlsx_file_pth: {excel_file_path}")
    print(f"xlsx path: {excel_file_path}\njson path: {json_path}")
    return example_list, example_list_pth


class EmbeddingStockBuilder(BaseStockBuilder):
    def __init__(self, stock_build_configs, sav_type='npy', **kwargs):
        self.sav_type = sav_type
        self.examples_list = load_json_file(stock_build_configs.get('examples_list_pth'))
        self.embedding_model = stock_build_configs.get('embedding_model')
        self.demonstration_list, self.demonstration_json_pth = \
            demonstration_backup(
                demonstration_pth=stock_build_configs.get('examples_list_pth'),
                sav_dir=stock_build_configs.get('sav_dir'),
                prefix=stock_build_configs.get('prefix'),
                eval_key_list=stock_build_configs.get('eval_key_list'),
            )
        self.search_key = stock_build_configs.get('search_key')
        super().__init__(**kwargs)

    def get_example_embeddings(self):
        """
        :param example_list: list of dict, each is an example.
        :param search_key: str, the key to search embedding
        :param embedding_model: the model to get the embedding
        :return: List of vector
        """
        text_list = organize_text_4_embedding(example_list=self.demonstration_list, search_key=self.search_key)

        example_embeddings = get_embedding(text_list, embedding_model=self.embedding_model)

        if self.embedding_model == "text_embedding_v1" or self.embedding_model == "text_embedding_v2":
            query_embedding_list = [item['embedding'] for item in example_embeddings.output['embeddings']]
        else:
            query_embedding_list = [item['embedding'] for item in example_embeddings['output']['embeddings']]
        return query_embedding_list

    def build_example_stock(self, cur_time=None) -> str:
        """
        get the embeddings of the content in the "search_key" of the example in the example_list,
        and sav it to f'{prefix}_examples_ver_{cur_time}.{sav_type}'
        """
        query_embedding_list = self.get_example_embeddings()
        if cur_time is not None:
            pass
        else:
            cur_time = get_current_date()

        embedding_array = np.vstack(query_embedding_list)

        embedding_sav_pth = os.path.join(self.sav_dir,
                                         f'{self.prefix}_emb_model:'
                                         f'<{self.embedding_model}>_search_key:'
                                         f'{self.search_key}_examples_ver_{cur_time}.npy')

        np.save(embedding_sav_pth, embedding_array)
        return embedding_sav_pth

    def update_example(self):


        cur_time = get_current_date()

        embedding_sav_pth = self.build_example_stock(example_list=example_list, search_key=search_key, sav_dir=sav_dir,
                                                     sav_type=self.sav_type,
                                                     prefix=prefix,
                                                     embedding_model=embedding_model,
                                                     cur_time=cur_time)
        update_icl_configs(config_pth=icl_config_pth,
                           embedding_pth=embedding_sav_pth,
                           embedding_model=embedding_model,
                           examples_list_pth=example_list_pth,
                           search_key=search_key)
        return None
