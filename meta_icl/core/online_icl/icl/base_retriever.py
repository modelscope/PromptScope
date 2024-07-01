from typing import List, Dict
from abc import ABC, abstractmethod

from meta_icl.core.utils.utils import sample_elements_and_ids, random_selection_method
from meta_icl.core.utils.sys_prompt_utils import (get_embedding, find_top_k_embeddings, message_formatting,
                                                  call_llm_with_message)
from meta_icl.core.utils.utils import load_file

import bm25s


class BaseRetriever(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def topk_selection(self):
        pass


class BM25Retriever(BaseRetriever):
    def __init__(self, example_list=None, bm25_index_pth=None, **kwargs):
        self.example_list = example_list
        assert (bm25_index_pth is not None) or (example_list is not None), ("either example_list or bm25_index_pth "
                                                                            "must be provided.")
        if bm25_index_pth is not None:
            self.retriever = bm25s.BM25.load(bm25_index_pth, load_corpus=True)

        if kwargs.get("stemmer_algo", None) is not None:
            self.stemmer = bm25s.Stemmer(algo=kwargs["stemmer_algo"])
        else:
            self.stemmer = None

        super().__init__()

    def topk_selection(self, query: str, num: int):
        """
        """
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        results, scores = reloaded_retriever.retrieve(query_tokens, k=num)


class CosineSimilarityRetriever(BaseRetriever):
    def __init__(self, embeddings: List, example_list=None):
        self.embeddings = embeddings
        self.example_list = example_list
        super().__init__()

    def topk_selection(self, query_embedding: List, num: int):
        """

        :param query_embedding: embedding vector
        :param num: int, the number of selection
        :return: {"selection_idx": list, "selection_score": list}
        """
        selection_results = find_top_k_embeddings(query_embedding=query_embedding,
                                                  list_embeddings=self.embeddings, k=num)
        selection_idx = [item[0] for item in selection_results]
        selection_score = [item[2] for item in selection_results]
        return {"selection_idx": selection_idx, "selection_score": selection_score}

    def get_examples(self, selection_ids: List) -> List:
        """

        :param selection_ids: list of idx
        :return: the examples selected
        """
        if self.example_list is not None:
            return [self.example_list[idx] for idx in selection_ids]
        else:
            ValueError("example_list is None, please provide the example list!")
