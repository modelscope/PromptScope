from meta_icl.core.online_icl.icl.ICL import BM25ICL

from meta_icl.core.online_icl.icl import EmbeddingICL
from meta_icl.core.utils import get_single_embedding, organize_text_4_embedding
from meta_icl.core.utils import message_formatting, call_llm_with_message
from meta_icl.contribs.app_main_followup.prompt.prompt_4_icl_followups import (formatting_str_type_main_chat,
                                                                               formatting_answer_out,
                                                                               formatting_multimodal_type_main_chat)
import json


class AppMainFollowupBM25(BM25ICL):
    def __init__(self, icl_configs, task_configs, **kwargs):
        """

        :param base_model: the base model to generate the intention analysis results.
        currently available choices: "Qwen_200B", "Qwen_70B", and "Qwen_14B"
        :param embedding_pth: the path storing the embedding vectors of the examples
        :param examples_pth: the path of the examples
        :param embedding_model: the model to get the embedding.
        currently only dashscope embedding model is available: "text_embedding_v1"
        """
        base_model = task_configs["base_model"]
        BM25_index_pth = icl_configs["BM25_index_pth"]
        examples_pth = icl_configs["examples_pth"]
        retriever_key_list = icl_configs["retriever_key_list"]

        super().__init__(base_model=base_model,
                         BM25_index_pth=BM25_index_pth,
                         examples_pth=examples_pth,
                         retriever_key_list=retriever_key_list,
                         task_configs=task_configs)


def get_BM25_followup_results(cur_query: dict,
                         task_configs: dict,
                         icl_configs: dict,
                         file_type="no", **kwargs):
    """
    """
    if file_type.lower() == "no":
        formatting_function = formatting_str_type_main_chat
    else:
        formatting_function = formatting_multimodal_type_main_chat
    followup_generator = AppMainFollowupBM25(icl_configs=icl_configs,
                                             task_configs=task_configs)
    results = followup_generator.get_results(
        cur_query,
        formatting_function=formatting_function,
        num=icl_configs["topk"],  **kwargs
    )
    print(results)
    results = formatting_answer_out(results)
    return results
