from typing import List, Dict

from meta_icl.utils.utils import sample_elements_and_ids, random_selection_method
from meta_icl.utils.sys_prompt_utils import get_embedding, find_top_k_embeddings
from meta_icl.utils.utils import load_file


class BaseRetrive():
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

    def get_examples(self, selection_ids: List) -> List:
        """

        :param selection_ids: list of idx
        :return: the examples selected
        """
        if self.example_list is not None:
            return [self.example_list[idx] for idx in selection_ids]
        else:
            ValueError("example_list is None, please provide the example list!")


class TopKRetriever(BaseRetrive):
    def __init__(self, embeddings: List, example_list=None):
        super().__init__(embeddings=embeddings, example_list=example_list)

class BaseICL:
    def __init__(self,
                 opt_model,
                 embedding_pth,
                 examples_pth,
                 embedding_model=""):
        self.opt_model = opt_model
        self._get_example_embeddings(embedding_pth)
        self._get_example_list(examples_pth)
        self.example_selector = BaseRetrive(example_list=self.examples,
                                            embeddings=self.embeddings)

    def _get_example_embeddings(self, embedding_pth):
        self.embeddings = load_file(embedding_pth)

    def _get_example_list(self, examples_pth):
        self.examples = load_file(examples_pth)

    def get_meta_prompt(self, query: str, num=5):
        pass

    def get_opt_prompt(self, query: str, num=5):
        pass




