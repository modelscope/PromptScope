from typing import List, Dict

from meta_icl.utils.utils import sample_elements_and_ids, random_selection_method
from meta_icl.utils.sys_prompt_utils import (get_embedding, find_top_k_embeddings, message_formatting,
                                             call_llm_with_message)
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


class CustomizedICL(BaseICL):
    def __init__(self, base_model,
                 embedding_pth,
                 examples_pth,
                 embedding_model=None,
                 configs=None
                 ):
        """

        :param base_model: the base model to generate the intention analysis results.
        currently available choices: "Qwen_200B", "Qwen_70B", and "Qwen_14B"
        :param embedding_pth: the path storing the embedding vectors of the examples
        :param examples_pth: the path of the examples
        :param embedding_model: the model to get the embedding.
        currently only dashscope embedding model is available: "text_embedding_v1"
        """
        super().__init__(opt_model=base_model,
                         embedding_pth=embedding_pth,
                         examples_pth=examples_pth)
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = "text_embedding_v1"

        self.task_configs = configs

    def get_meta_prompt(self, cur_query: dict, embedding_key: list, num=3, formatting_function=None):
        """
        :param cur_query: the query to generate the intention analysis results.
        :param embedding_key: the key to get the embedding.
        :param num: the number of selection
        :param formatting_function: the formatting function to generate the final query

        :return: the meta prompt
        """
        try:
            test_to_query_embedding = organize_text_4_embedding(example_list=[cur_query],
                                                                search_key=embedding_key)
            query_embedding = get_single_embedding(test_to_query_embedding, embedding_model=self.embedding_model)
            selection_results = self.example_selector.topk_selection(query_embedding=query_embedding, num=num)
            print(f"selection_results: {selection_results}")
        except:
            # If failed to select samples, return the default first three samples.
            print("error in getting the query embedding. Use the default index")
            selection_results = {"selection_idx": [0, 1, 2]}

        selection_examples = self.example_selector.get_examples(selection_results["selection_idx"])
        print(f"selection_examples: {selection_examples}")
        query = formatting_function(selection_examples, cur_query, configs=self.task_configs)
        return query

    def get_results(self, cur_query: dict, embedding_key: list, num=3, formatting_function=None):
        query = self.get_meta_prompt(cur_query=cur_query,
                                     num=num, embedding_key=embedding_key,
                                     formatting_function=formatting_function)
        print(query)
        message = message_formatting(system_prompt='You are a helpful assistant', query=query)
        res = call_llm_with_message(messages=message, model=self.opt_model)
        print(res)
        return res
