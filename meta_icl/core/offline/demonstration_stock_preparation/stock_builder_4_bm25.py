import os.path

import bm25s
from meta_icl.core.utils.utils import organize_text_4_embedding, get_current_date
from meta_icl.core.utils.sys_prompt_utils import load_json_file
from meta_icl.core.utils.retrieve_utils import STOPWORDS_EN, STOPWORDS_ZH, demonstration_backup
import Stemmer
from meta_icl.core.offline.demonstration_stock_preparation.base_stock_builder import BaseStockBuilder
from meta_icl.core.utils.config_utils import update_icl_configs_BM25, load_config
from loguru import logger


def get_demonstration_corpus(examples_list, search_key):
    return organize_text_4_embedding(examples_list, search_key)


class BM25StockBuilder(BaseStockBuilder):
    def __init__(self, stock_build_configs, **kwargs):
        self.bm25_index_pth = None
        self.search_key_list = stock_build_configs.get('search_key')
        self.example_list_pth = stock_build_configs.get("examples_list_pth")
        self._load_demonstration_list(self.example_list_pth)
        self.corpus = get_demonstration_corpus(self.examples_list, self.search_key_list)

        self.online_icl_pth = stock_build_configs.get('icl_config_pth')
        self.bm25_index_dir = stock_build_configs.get('sav_dir')
        self.bm25_index_pth_prefix = stock_build_configs.get('prefix')

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
        self.examples_list = load_json_file(demonstration_json_pth)

    def _update_bm25_index_pth(self, bm25_index_pth):
        self.bm25_index_pth = bm25_index_pth

    def build_bm25_stock(self):

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
        update_icl_configs_BM25(config_pth=self.online_icl_pth,
                                examples_list_pth=self.example_list_pth,
                                search_key=self.search_key_list, BM_25_index_dir=self.bm25_index_pth)

    def build_stock(self):
        demonstration_backup(sav_dir=self.bm25_index_dir, demonstration_pth=self.example_list_pth,
                             prefix=self.bm25_index_pth)
        bm25_index_pth = self.build_bm25_stock()
        self._update_bm25_index_pth(bm25_index_pth)
        self.update_icl_config()
def prepare_BM25_stock(stock_builder_config_pth: str):
    stock_builder_configs = load_config(config_pth=stock_builder_config_pth, as_edict=False)
    embedding_stock_builder = BM25StockBuilder(stock_builder_configs)
    embedding_stock_builder.build_stock()


if __name__ == '__main__':
    stock_builder_config_pth = "conf/agent_followup_configs/stock_builder_configs/bm25_demonstration_stock_config.yaml"
    prepare_BM25_stock(stock_builder_config_pth)
#
# if __name__ == '__main__':
#     pth = "data/icl_app_mainchat_followup/main_chat_str_icl_examples_ver_2024-06-05 22:34:25.json"
#     stock_pth = "data/icl_bm25_demo"
#     stock_builder = BM25StockBuilder(
#         demonstration_json_pth=pth,
#         search_key_list=["chat_history", "last_query"])
#     stock_builder.build_bm25_stock(stock_pth)
