from meta_icl.core.online_icl.icl.ICL import BM25ICL

from meta_icl.core.online_icl.icl import EmbeddingICL
from meta_icl.core.utils import get_single_embedding, organize_text_4_embedding, timer
from meta_icl.core.utils import message_formatting, call_llm_with_message
from meta_icl.contribs.app_main_followup.prompt.prompt_4_icl_followups import (formatting_str_type_main_chat,
                                                                               formatting_answer_out,
                                                                               formatting_multimodal_type_main_chat)
import json



class AppMainFollowupBM25(BM25ICL):
    def __init__(self, icl_configs, task_configs):
        """

        :param icl_configs: configs for ICL
        :param task_configs: configs for task
        Examples: {
    "icl_configs": {
        "BM25_index_pth": "data/icl_bm25_demo/demonstrations_2024-07-02 12:09:11",
        "examples_pth": "data/icl_app_mainchat_followup/main_chat_str_icl_examples_ver_2024-06-05 22:34:25.json",
        "topk": 3,
        "retriever_key_list": [
            "chat_history",
            "last_query"
        ]
    },
    "task_configs": {
        "base_model": "Qwen_70B",
        "num_questions": 3
    }
}
        """
        base_model = task_configs.get("base_model")
        BM25_index_pth = icl_configs.get("BM_25_index_dir")
        examples_pth = icl_configs.get("examples_list_pth")
        retriever_key_list = icl_configs.get("retriever_key_list")

        super().__init__(base_model=base_model,
                         BM25_index_pth=BM25_index_pth,
                         examples_pth=examples_pth,
                         retriever_key_list=retriever_key_list,
                         task_configs=task_configs)


@timer
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

    BM25_retriever_configs = icl_configs.get("BM25_retriever_configs")
    followup_generator = AppMainFollowupBM25(icl_configs=BM25_retriever_configs,
                                             task_configs=task_configs)
    num_selection = BM25_retriever_configs.get("topk", 3)
    results = followup_generator.get_results(
        cur_query,
        formatting_function=formatting_function,
        num=num_selection, **kwargs
    )
    print(results)
    results = formatting_answer_out(results)
    return results
