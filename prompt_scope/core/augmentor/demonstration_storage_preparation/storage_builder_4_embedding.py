import os

import numpy as np

from prompt_scope.core.offline.demonstration_storage_preparation.storage_builder_4_bm25 import BaseStorageBuilder
from prompt_scope.core.utils.config_utils import load_config, update_icl_configs_embedding
from prompt_scope.core.utils.retrieve_utils import demonstration_backup
from prompt_scope.core.utils.sys_prompt_utils import load_json_file, get_embedding
from prompt_scope.core.utils.utils import (get_current_date,
                                       organize_text_4_embedding)


class EmbeddingStorageBuilder(BaseStorageBuilder):
    """
    Embedding storage builder class for constructing storage solutions based on provided configurations.
    """

    def __init__(self, storage_build_configs, sav_type='npy', **kwargs):
        """
        :param storage_build_configs: Dictionary containing configuration parameters required for building the storage solution.
        :param sav_type: Storage type, default is 'npy'. Determines the file format for storage.
        :param **kwargs: Additional keyword arguments to be passed to the parent class constructor.

        Attributes:
        - sav_type: Storage type, prioritizes the value from the configuration dictionary or uses the default if not specified.
        - examples_list: List of examples loaded from the path specified in the configuration.
        - embedding_model: Name of the embedding model from the configuration.
        - embedding_shape: Shape configuration of the embedding model.
        - demonstration_list: List of demonstrations containing a series of demonstration data.
        - demonstration_json_pth: JSON path for the demonstration data.
        - demo_storage_name_prefix: Prefix for the demonstration storage name.
        - demo_storage_sav_dir: Directory for saving the demonstration storage.
        - search_key: Search key.
        - online_icl_pth: Path to the online ICL configuration.

        """
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
        Get the embeddings of the content in the "search_key" of the examples in the example list,
        and save them to `f'{prefix}_examples_ver_{cur_time}.{sav_type}'`.

        :param cur_time: Optional, current time used to determine the version number of the saved file.
                    If not provided, the current date is used.

        :return The path of the saved embedding file.
        """
        # Get the example embedding list
        query_embedding_list = self.get_example_embeddings()

        # If cur_time is not provided, use the current date as the default value
        if cur_time is not None:
            pass
        else:
            cur_time = get_current_date()

        # Process embeddings based on the save type
        if self.sav_type == 'npy':
            # Convert the embedding list to a NumPy array for saving as an npy file
            embedding_array = np.vstack(query_embedding_list)
            embedding_sav_pth = os.path.join(self.demo_storage_sav_dir,
                                             f'{self.demo_storage_name_prefix}_emb_model:'
                                             f'<{self.embedding_model}>_search_key:'
                                             f'{self.search_key}_examples_ver_{cur_time}.npy')
            # Save the embedding array to an npy file
            np.save(embedding_sav_pth, embedding_array)

        elif self.sav_type == 'idx':
            # Use the Faiss library to create an index and save it as an idx file
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
        """
        Build storage by generating a new embedding save path based on the current time and updating configurations.

        This method first retrieves the current date and time to generate an embedding save path.
        After the path is generated, it updates the internal embedding save path using the `_update_embedding_sav_pth` method.
        Finally, it calls the `update_icl_configs_embedding` function to update the embedding configuration with the new embedding path,
        embedding model, examples list path, and search key.
        """
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
    """
    Prepare the embedding storage. Load the storage builder configuration and build the embedding storage.

    :param storage_builder_config_pth (str): The path to the storage builder configuration.
    """
    storage_builder_configs = load_config(config_pth=storage_builder_config_pth, as_edict=False)
    embedding_storage_builder = EmbeddingStorageBuilder(storage_builder_configs)
    embedding_storage_builder.build_storage()
