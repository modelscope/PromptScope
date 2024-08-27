from typing import List, Dict
from abc import ABC, abstractmethod

from meta_icl.core.utils.utils import sample_elements_and_ids, random_selection_method
from meta_icl.core.utils.sys_prompt_utils import (get_embedding, find_top_k_embeddings, message_formatting,
                                                  call_llm_with_message)
from meta_icl.core.utils.utils import load_file, organize_text_4_embedding, get_single_embedding
from meta_icl.core.online_icl.icl.base_retriever import CosineSimilarityRetriever, BM25Retriever, FaissRetriever
from meta_icl.core.models.generation_model import GenerationModel
from loguru import logger
import time
from meta_icl import CONFIG_REGISTRY


class BaseICL(ABC):
    @abstractmethod
    def get_meta_prompt(self):
        pass

    @abstractmethod
    def _load_demonstration_selector(self):
        pass

    @abstractmethod
    def get_meta_prompt(self, query: str, num: int, **kwargs):
        pass

    @abstractmethod
    def get_results(self, **kwargs):
        pass

    def _update_formatting_function(self):
        pass


class BM25ICL(BaseICL):
    def __init__(self,
                 base_model,
                 BM25_index_pth,
                 examples_pth,
                 retriever_key_list,
                 task_configs=None):
        self.BM25_index_pth = BM25_index_pth
        self._load_demonstration_list(examples_pth)
        self._load_demonstration_selector()
        self.retriever_key_list = retriever_key_list
        if task_configs is not None:
            self.task_configs = task_configs
        else:
            self.task_configs = CONFIG_REGISTRY.module_dict['task_configs']
        if base_model is not None:
            self.base_model = base_model
        else:
            self.base_model = AioGenerationModel(**self.task_configs["model_config"]["generation"])

    def _load_demonstration_selector(self):
        self.example_selector = BM25Retriever(example_list=self.example_list,
                                              bm25_index_pth=self.BM25_index_pth)

    def _load_demonstration_list(self, examples_pth):
        self.example_list = load_file(examples_pth)

    def get_meta_prompt(self, cur_query: dict, formatting_function, num=3, **kwargs):
        """
        :param cur_query: the query to generate the intention analysis results.
        :param search_key_list: the key to index & search by bm25.
        :param num: the number of selection
        :param formatting_function: the formatting function to generate the final query
        :return: the meta prompt
        """
        query_to_search = organize_text_4_embedding(example_list=[cur_query],
                                                    search_key=self.retriever_key_list)
        logger.info(f"query to search: {query_to_search}")
        selection_results = self.example_selector.topk_selection(query=query_to_search[0], num=num)
        print(f"selection_results: {selection_results}")
        selection_examples = self.example_selector.get_examples(selection_results["selection_idx"])
        print(f"selection_examples: {selection_examples}")
        query = formatting_function(selection_examples, cur_query, configs=self.task_configs)
        return query

    def get_results(self, cur_query: dict, formatting_function, num=3, **kwargs):
        query = self.get_meta_prompt(cur_query=cur_query,
                                     num=num,
                                     formatting_function=formatting_function)
        print(query)
        message = message_formatting(system_prompt='You are a helpful assistant', query=query)
        res = call_llm_with_message(messages=message, model=self.base_model, **kwargs)
        print(res)
        return res


class EmbeddingICL(BaseICL):
    def __init__(self, base_model=None,
                 embedding_pth=None,
                 examples_pth=None,
                 embedding_model=None,
                 task_configs=None,
                 retriever_key_list: List = None
                 ):
        """

        :param base_model: the base model to generate the intention analysis results.
        currently available choices: "Qwen_200B", "Qwen_70B", and "Qwen_14B"
        :param embedding_pth: the path storing the embedding vectors of the examples
        :param examples_pth: the path of the examples
        :param embedding_model: the model to get the embedding.
        currently only dashscope embedding model is available: "text_embedding_v1"
        """
        super().__init__()
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = "text_embedding_v1"
        if task_configs is not None:
            self.task_configs = task_configs
        else:
            self.task_configs = CONFIG_REGISTRY.module_dict['task_configs']
        if base_model is not None:
            self.base_model = base_model
        else:
            self.base_model = AioGenerationModel(**self.task_configs["model_config"]["generation"])
        self._get_example_embeddings(embedding_pth)
        self._get_example_list(examples_pth)
        self._load_demonstration_selector(sav_type=embedding_pth.split('.')[-1])
        self.retriever_key_list = retriever_key_list

    def _load_demonstration_selector(self, sav_type='npy'):
        if sav_type == 'npy':
            self.example_selector = CosineSimilarityRetriever(example_list=self.examples,
                                                              embeddings=self.embeddings)
        elif sav_type == 'index':
            self.example_selector = FaissRetriever(example_list=self.examples,
                                                   index=self.embeddings)

    def _get_example_embeddings(self, embedding_pth):
        self.embeddings = load_file(embedding_pth)

    def _get_example_list(self, examples_pth):
        self.examples = load_file(examples_pth)

    def get_meta_prompt(self, cur_query: dict, formatting_function:object=None, num=3):
        """
        :param cur_query: the query to generate the intention analysis results.
        :param embedding_key: the key to get the embedding.
        :param num: the number of selection
        :param formatting_function: the formatting function to generate the final query

        :return: the meta prompt
        """


        try:
            query_embedding = get_single_embedding([cur_query], embedding_model=self.embedding_model,
                                                   search_key=self.retriever_key_list)
            selection_results = self.example_selector.topk_selection(query_embedding=query_embedding, num=num)
            logger.info(f"selection_results: {selection_results}")
        except Exception as e:
            # If failed to select samples, return the default first three samples.
            logger.error(e)
            logger.error("error in getting the query embedding. Use the default index")
            selection_results = {"selection_idx": [0, 1, 2]}

        selection_examples = self.example_selector.get_examples(selection_results["selection_idx"])
        logger.info(f"selection_examples: {selection_examples}")
        query = formatting_function(selection_examples, cur_query, configs=self.task_configs)
        return query

    def get_results(self, cur_query: dict, formatting_function, num=3, **kwargs):

        query = self.get_meta_prompt(cur_query=cur_query,
                                     num=num,
                                     formatting_function=formatting_function)
        logger.info(f"query: {query}")
        message = message_formatting(system_prompt='You are a helpful assistant', query=query)
        res = call_llm_with_message(messages=message, model=self.base_model, **kwargs)
        # res = self.base_model.call(messages=message).message.content
        logger.info(f"res: {res}")
        return res
