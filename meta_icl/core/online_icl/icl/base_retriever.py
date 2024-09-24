from typing import List, Dict, Any
from abc import ABC, abstractmethod

from meta_icl.core.utils.utils import sample_elements_and_ids, random_selection_method
from meta_icl.core.utils.sys_prompt_utils import (get_embedding, find_top_k_embeddings,
                                                  message_formatting,
                                                  call_llm_with_message)
from meta_icl.core.utils.utils import load_file

import bm25s


class BaseRetriever(ABC):
    """
    Defines an abstract base class `BaseRetriever` that should be implemented by retriever classes.
    """
    def __init__(self):
        pass

    @abstractmethod
    def topk_selection(self, query: str, num: int) -> Dict[str, List]:
        """
        Selects the top-K items based on a query string.

        :param query (str): The query string.
        :param num (int): The number of top-K items to return.

        :return Dict[str, List]: A dictionary containing the top-K selected items based on the query string.
        """
        pass

    @abstractmethod
    def get_examples(self, selection_ids: List) -> List:
        """
        Retrieves examples based on selected IDs.

        :param selection_ids (List): A list of selected item IDs.

        :return List: A list containing the examples retrieved based on `selection_ids`.
        """
        pass


class BM25Retriever(BaseRetriever):
    """
    BM25 retriever that inherits from BaseRetriever and implements text retrieval using the BM25 algorithm.

    :param example_list: A list of examples containing documents or question-answer pairs used to build the BM25 index.
    :param bm25_index_pth: The path to a precomputed BM25 index. If provided, the index will be loaded from this path.
    :param **kwargs: Additional keyword arguments, such as stemmer algorithm configuration.

    Description:
    The constructor initializes the BM25 retriever. It sets up the internal state based on either the provided `example_list`
    or the loaded `bm25_index_pth`. If `example_list` is provided, it generates the BM25 index from these examples.
    If `bm25_index_pth` is provided, it loads the precomputed index. Additionally, it initializes the stemmer based
    on the `stemmer_algo` parameter in `kwargs`.
    """
    def __init__(self, example_list=None, bm25_index_pth=None, **kwargs):
        # Initialize example_list, ensuring that either example_list or bm25_index_pth is provided
        self.example_list = example_list
        assert (bm25_index_pth is not None) or (example_list is not None), ("either example_list or bm25_index_pth "
                                                                            "must be provided.")
        # If bm25_index_pth is provided, load the BM25 index from the specified path
        if bm25_index_pth is not None:
            self.retriever = bm25s.BM25.load(bm25_index_pth, load_corpus=True)

        # Initialize the stemmer based on the stemmer_algo parameter in kwargs
        if kwargs.get("stemmer_algo", None) is not None:
            self.stemmer = bm25s.Stemmer(algo=kwargs["stemmer_algo"])
        else:
            self.stemmer = None

        super().__init__()

    def topk_selection(self, query: str, num: int) -> Dict:
        """
        Selects the top-k items based on the query string.

        :param query (str): The query string.
        :param num (int): The number of items to select.

        :return Dict: A dictionary containing the selected indices and scores.
        """
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        results, scores = self.retriever.retrieve(query_tokens, k=num)
        results = [element for row in results for element in row]
        scores = [element for row in scores for element in row]
        print(f"results: {results}\ntype: {type(scores)}\nscores: {scores}, type: {type(scores)}")
        selection_idx = [item["id"] for item in results]

        # selected_examples = [item["text"] for item in results]

        return {
            "selection_idx": selection_idx,
            "selection_score": scores,
            # "selected_examples": selected_examples
        }

    def get_examples(self, selection_ids: List) -> List:
        """
        :param selection_ids: list of idx
        :return: the examples selected
        """
        if self.example_list is not None:
            return [self.example_list[idx] for idx in selection_ids]
        else:
            ValueError("example_list is None, please provide the example list!")


class CosineSimilarityRetriever(BaseRetriever):
    """
    A retriever class that uses cosine similarity to find the most similar examples.
    Inherits from BaseRetriever.
    """
    def __init__(self, embeddings: List, example_list=None):
        """
        Initializes the CosineSimilarityRetriever class.

        :param embeddings: A list of embedding vectors.
        :param example_list: A list of examples corresponding to the embeddings.
        """
        self.embeddings = embeddings
        self.example_list = example_list
        super().__init__()

    def topk_selection(self, query_embedding: List, num: int):
        """
        Selects the top-k most similar embeddings to the query embedding.

        :param query_embedding: embedding vector
        :param num: int, the number of selection
        :return: {"selection_idx": list, "selection_score": list}
        """
        selection_results = find_top_k_embeddings(query_embedding=query_embedding,
                                                  list_embeddings=self.embeddings, k=num)
        selection_idx = [item[0] for item in selection_results]
        selection_score = [item[2] for item in selection_results]

        return {"selection_idx": selection_idx,
                "selection_score": selection_score}

    def get_examples(self, selection_ids: List) -> List:
        """
        Gets the examples corresponding to the selected embedding indices.

        :param selection_ids: list of idx
        :return: the examples selected
        """
        if self.example_list is not None:
            return [self.example_list[idx] for idx in selection_ids]
        else:
            ValueError("example_list is None, please provide the example list!")

class FaissRetriever(BaseRetriever):
    def __init__(self, index: Any, example_list=None):
        self.index = index
        self.example_list = example_list
        super().__init__()

    def topk_selection(self, query_embedding: List, num: int):
        """

        :param query_embedding: embedding vector
        :param num: int, the number of selection
        :return: {"selection_idx": list, "selection_score": list}
        """
        import numpy as np
        D, I = self.index.search(np.array(query_embedding).reshape(1, -1), num)
        return {"selection_idx": I[0],
                "selection_score": [1/x for x in D]}
    
    def get_examples(self, selection_ids: List) -> List:
        """

        :param selection_ids: list of idx
        :return: the examples selected
        """
        if self.example_list is not None:
            return [self.example_list[idx] for idx in selection_ids]
        else:
            ValueError("example_list is None, please provide the example list!")
