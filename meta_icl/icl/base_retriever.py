from typing import List, Dict

from meta_icl.utils.utils import sample_elements_and_ids, random_selection_method
from meta_icl.utils.sys_prompt_utils import get_embedding, find_top_k_embeddings


class BaseRetriever:
    def __init__(self, embeddings: List, example_list=None):
        self.embeddings = embeddings
        self.example_list = example_list

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

    def get_examples(self, selection_ids: List):
        """

        :param selection_ids: list of idx
        :return: the examples selected
        """
        if self.example_list is not None:
            return [self.example_list[idx] for idx in selection_ids]
        else:
            ValueError("example_list is None, please provide the example list!")
