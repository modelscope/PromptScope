import os.path

import bm25s
from meta_icl.core.utils.utils import organize_text_4_embedding, get_current_date
from meta_icl.core.utils.sys_prompt_utils import load_json_file
from meta_icl.core.utils.retrieve_utils import STOPWORDS_EN, STOPWORDS_ZH
import Stemmer
from meta_icl.core.offline.demonstration_stock_preparation.base_stock_builder import BaseStockBuilder


def get_demonstration_corpus(examples_list, search_key):
    return organize_text_4_embedding(examples_list, search_key)


class BM25StockBuilder(BaseStockBuilder):
    def __init__(self, demonstration_json_pth, search_key_list, **kwargs):
        self._load_demonstration_list(demonstration_json_pth)
        self.corpus = get_demonstration_corpus(self.examples_list, search_key_list)
        self.search_key_list = search_key_list
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
            self.stopwords.extend(search_key_list)

    def _load_demonstration_list(self, demonstration_json_pth):
        self.examples_list = load_json_file(demonstration_json_pth)

    def build_stock(self, bm25_stock_pth, prefix=""):
        bm_25_index_pth = os.path.join(bm25_stock_pth, f"{prefix}_demonstrations_{get_current_date()}")
        corpus_tokens = bm25s.tokenize(self.corpus, stopwords=self.stopwords, stemmer=self.stemmer)

        # Create the BM25 model and index the corpus
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        retriever.save(bm_25_index_pth, corpus=self.corpus)
        return bm_25_index_pth


if __name__ == '__main__':
    pth = "data/icl_app_mainchat_followup/main_chat_str_icl_examples_ver_2024-06-05 22:34:25.json"
    stock_pth = "data/icl_bm25_demo"
    stock_builder = BM25StockBuilder(
        demonstration_json_pth=pth,
        search_key_list=["chat_history", "last_query"]
    )
    stock_builder.build_stock(stock_pth)
