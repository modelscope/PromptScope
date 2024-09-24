import os.path

import bm25s
from meta_icl.core.utils.utils import organize_text_4_embedding, get_current_date
from meta_icl.core.utils.sys_prompt_utils import load_json_file
from meta_icl.core.utils.retrieve_utils import STOPWORDS_EN, STOPWORDS_ZH, demonstration_backup
import Stemmer
from meta_icl.core.offline.demonstration_storage_preparation.base_storage_builder import BaseStorageBuilder
from meta_icl.core.utils.config_utils import update_icl_configs_BM25, load_config
from loguru import logger
from typing import List


def get_demonstration_corpus(examples_list: List, search_key: List)->List:
    """
    Generates a demonstration corpus using the provided examples and search keys.

    This function organizes text data for embedding processing. It takes an examples list and search keys as input,
    and returns a list of organized text data, which is processed by the `organize_text_4_embedding` function.

    :param examples_list (List): A list of text examples to be processed.
    :param search_key (List): A list of search keys used for organizing the text.

    :return List: A list of organized text data, ready for embedding processing.
    """
    return organize_text_4_embedding(examples_list, search_key)


class BM25StorageBuilder(BaseStorageBuilder):
    """
    BM25 Storage Builder class for constructing and managing the generation and updating of BM25 indexes.

    """
    def __init__(self, storage_build_configs, **kwargs):
        """
        :param storage_build_configs: A dictionary containing storage build configurations such as index paths, example list paths, etc.
        :param **kwargs: Variable keyword arguments to pass additional configurations like stemming algorithm and stopword lists.

        Attributes:
        - bm25_index_pth: File path of the BM25 index.
        - search_key_list: List of search keywords.
        - example_list_pth: File path of the example list.
        - corpus: Corpus of example documents.
        - online_icl_pth: Path to the online incontext learning configuration.
        - bm25_index_dir: Directory for the BM25 index.
        - bm25_index_pth_prefix: Prefix for the BM25 index file path.
        - stemmer: Stemmer instance.
        - stopwords: Stopword list.
        """
        # Initialize BM25 index path and keyword list
        self.bm25_index_pth = None
        self.search_key_list = storage_build_configs.get('search_key')
        self.example_list_pth = storage_build_configs.get("examples_list_pth")

        # Load example list
        self._load_demonstration_list(self.example_list_pth)

        # Build example corpus
        self.corpus = get_demonstration_corpus(self.examples_list, self.search_key_list)

        self.online_icl_pth = storage_build_configs.get('icl_config_pth')
        self.bm25_index_dir = storage_build_configs.get('sav_dir')
        self.bm25_index_pth_prefix = storage_build_configs.get('prefix')

        # Initialize stemmer (if an algorithm type is provided)
        if kwargs.get("stemmer_algo", None) is not None:
            self.stemmer = Stemmer.Stemmer(kwargs.get("stemmer_algo"))
        else:
            self.stemmer = None
        if kwargs.get("stopwords", None) is not None:
            self.stopwords = kwargs.get("stopwords")
        else:
            self.stopwords = []
            self.stopwords.extend(STOPWORDS_EN)
            self.stopwords.extend(STOPWORDS_ZH)
            self.stopwords.extend(self.search_key_list)

    def _load_demonstration_list(self, demonstration_json_pth):
        """
        Load the example list from a JSON file.

        :param demonstration_json_pth: File path of the example list JSON file.
               """
        self.examples_list = load_json_file(demonstration_json_pth)

    def _update_bm25_index_pth(self, bm25_index_pth):
        """
        Update the file path of the BM25 index.

        Parameters:
        - bm25_index_pth: New file path of the BM25 index.
        """
        # Update BM25 index path
        self.bm25_index_pth = bm25_index_pth

    def build_bm25_storage(self):
        """
        Builds and stores the BM25 index.

        This method tokenizes the corpus, creates a BM25 model, indexes the tokenized corpus, and stores the model along with the corpus text.

        :return str: The path where the BM25 index is stored.
        """
        # Tokenize the corpus

        corpus_tokens = bm25s.tokenize(self.corpus, stopwords=self.stopwords, stemmer=self.stemmer)
        bm25_index_pth = os.path.join(self.bm25_index_dir,
                                      f"{self.bm25_index_pth_prefix}_demonstrations_{get_current_date()}")

        # Create the BM25 model and index the corpus
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        retriever.save(bm25_index_pth, corpus=self.corpus)

        logger.info(f"bm25 index path is: {bm25_index_pth}")
        return bm25_index_pth

    def update_icl_config(self):
        """
        Updates the ICL configuration based on BM25.

        This method updates the ICL configuration using the specified config file path, example list path, and list of search keys, pointing to the new BM25 index directory.
        """
        update_icl_configs_BM25(config_pth=self.online_icl_pth,
                                examples_list_pth=self.example_list_pth,
                                search_key=self.search_key_list, BM_25_index_dir=self.bm25_index_pth)

    def build_storage(self):
        """
        Overall process control method for building storage.

        This method handles backing up the example list, building and storing the BM25 index, updating the BM25 index path, and updating the ICL configuration.
        """
        # Backup the example list
        demonstration_backup(sav_dir=self.bm25_index_dir, demonstration_pth=self.example_list_pth,
                             prefix=self.bm25_index_pth)
        bm25_index_pth = self.build_bm25_storage()
        self._update_bm25_index_pth(bm25_index_pth)
        self.update_icl_config()


def prepare_BM25_storage(storage_builder_config_pth: str)->None:
    """
    Prepares the BM25 storage.

    Loads configurations based on the provided storage builder configuration file path and creates an instance of the BM25 storage builder.

    :param storage_builder_config_pth (str): Path to the storage builder configuration file.
    """
    # Load storage builder configurations
    storage_builder_configs = load_config(config_pth=storage_builder_config_pth, as_edict=False)
    embedding_storage_builder = BM25StorageBuilder(storage_builder_configs)
    embedding_storage_builder.build_storage()

