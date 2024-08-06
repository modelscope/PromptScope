from meta_icl.core.offline.demonstration_storage_preparation.storage_builder_4_bm25 import BaseStorageBuilder
from meta_icl.core.utils.utils import (get_current_date,
                                       convert_xlsx_2_json,
                                       convert_json_2_xlx,
                                       organize_text_4_embedding,
                                       sav_yaml, load_yaml_file)
from meta_icl.core.utils.config_utils import load_config, update_icl_configs_embedding
from meta_icl.core.utils.sys_prompt_utils import load_json_file, get_embedding
import os
import numpy as np
from meta_icl.core.utils.retrieve_utils import demonstration_backup


class EmbeddingStorageBuilder(BaseStorageBuilder):
    def __init__(self, storage_build_configs, sav_type='npy', **kwargs):
        self.sav_type = storage_build_configs.get('sav_type') or sav_type
        self.examples_list = load_json_file(storage_build_configs.get('examples_list_pth'))
        self.embedding_model = storage_build_configs.get('embedding_model')
        self.embedding_shape = storage_build_configs.get('embedding_shape')
        self.demonstration_list, self.demonstration_json_pth = \
            demonstration_backup(
                demonstration_pth=storage_build_configs.get('examples_list_pth'),
                sav_dir=storage_build_configs.get('sav_dir'),
                prefix=storage_build_configs.get('prefix'),
                eval_key_list=storage_build_configs.get('eval_key_list'),
            )
        self.demo_storage_name_prefix = storage_build_configs.get('prefix')
        self.demo_storage_sav_dir = storage_build_configs.get('sav_dir')
        self.search_key = storage_build_configs.get('search_key')
        self.online_icl_pth = storage_build_configs.get('icl_config_pth')
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

    def build_example_storage(self, cur_time=None) -> str:
        """
        get the embeddings of the content in the "search_key" of the example in the example_list,
        and sav it to f'{prefix}_examples_ver_{cur_time}.{sav_type}'
        """
        query_embedding_list = self.get_example_embeddings()
        if cur_time is not None:
            pass
        else:
            cur_time = get_current_date()

        if self.sav_type == 'npy':
            embedding_array = np.vstack(query_embedding_list)
            embedding_sav_pth = os.path.join(self.demo_storage_sav_dir,
                                             f'{self.demo_storage_name_prefix}_emb_model:'
                                             f'<{self.embedding_model}>_search_key:'
                                             f'{self.search_key}_examples_ver_{cur_time}.npy')
            np.save(embedding_sav_pth, embedding_array)

        elif self.sav_type == 'idx':
            import faiss
            from faiss import write_index
            embedding_sav_pth = os.path.join(self.demo_storage_sav_dir,
                                             f'{self.demo_storage_name_prefix}_emb_model:'
                                             f'<{self.embedding_model}>_search_key:'
                                             f'{self.search_key}_examples_ver_{cur_time}.index')
            index = faiss.IndexFlatL2(self.embedding_shape)
            embedding_array = np.array(query_embedding_list).reshape(-1, self.embedding_shape)
            print(embedding_array.shape)
            index.add(embedding_array)
            write_index(index, embedding_sav_pth)

        return embedding_sav_pth

    def _update_embedding_sav_pth(self, embedding_sav_pth: str):
        self.embedding_sav_pth = embedding_sav_pth

    def build_storage(self):
        cur_time = get_current_date()
        embedding_sav_pth = self.build_example_storage(cur_time=cur_time)
        self._update_embedding_sav_pth(embedding_sav_pth=embedding_sav_pth)
        update_icl_configs_embedding(config_pth=self.online_icl_pth,
                                     embedding_pth=self.embedding_sav_pth,
                                     embedding_model=self.embedding_model,
                                     examples_list_pth=self.demonstration_json_pth,
                                     search_key=self.search_key)
        return None


def prepare_embedding_storage(storage_builder_config_pth: str):
    storage_builder_configs = load_config(config_pth=storage_builder_config_pth, as_edict=False)
    embedding_storage_builder = EmbeddingStorageBuilder(storage_builder_configs)
    embedding_storage_builder.build_storage()


if __name__ == '__main__':
    storage_builder_config_pth = "conf/agent_followup_configs/demonstration_embedding_storage_config.yaml"
    prepare_embedding_storage(storage_builder_config_pth)
    #
    # storage_build_configs = load_yaml_file(storage_builder_config_pth)
    #
    # storage_builder = EmbeddingStorageBuilder(storage_build_configs)
    # storage_builder.build_storage()
