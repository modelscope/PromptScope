from meta_icl.core.online_icl.icl import EmbeddingICL
from meta_icl.core.utils import get_single_embedding, organize_text_4_embedding
from meta_icl.core.utils import message_formatting, call_llm_with_message
from meta_icl.contribs.app_main_followup.prompt.prompt_4_icl_followups import (formatting_str_type_main_chat,
                                                                               formatting_answer_out,
                                                                               formatting_multimodal_type_main_chat)
import json


class AppMainFollowup(EmbeddingICL):
    def __init__(self, base_model,
                 embedding_pth,
                 examples_pth,
                 embedding_model=None,
                 task_config=None,
                 ):
        """

        :param base_model: the base model to generate the intention analysis results.
        currently available choices: "Qwen_200B", "Qwen_70B", and "Qwen_14B"
        :param embedding_pth: the path storing the embedding vectors of the examples
        :param examples_pth: the path of the examples
        :param embedding_model: the model to get the embedding.
        currently only dashscope embedding model is available: "text_embedding_v1"
        """
        super().__init__(base_model=base_model,
                         embedding_pth=embedding_pth,
                         examples_pth=examples_pth,
                         task_configs=task_config)
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = "text_embedding_v1"


def get_followup_results(cur_query: dict,
                         embedding_key: list,
                         base_model: str,
                         embedding_pth,
                         examples_pth,
                         embedding_model=None,
                         model_config=None,
                         task_config=None,
                         num=3,
                         file_type="no"):
    """

    :param cur_query: dict, {"previous": [{
      "优惠券用不了？", "让我查一下您的优惠券信息。"
    }],
    "last_query": "我要退款"}
    :param embedding_key: the key to request the embedding in cur_query.
    :param base_model: the model to generate the intention analysis results
    :param embedding_pth: the path storing the embedding vectors of the examples
    :param examples_pth: the path of the examples
    :param embedding_model: the model to get the embedding.
        currently only dashscope embedding model is available: "text_embedding_v1"
    :param num: the number of demonstration.md examples.
    :return: list of str, list of followup questions.
    """
    if file_type.lower() == "no":
        formatting_function = formatting_str_type_main_chat
    else:
        formatting_function = formatting_multimodal_type_main_chat
    followup_generator = AppMainFollowup(base_model=base_model,
                                         embedding_pth=embedding_pth,
                                         examples_pth=examples_pth,
                                         embedding_model=embedding_model,
                                         task_config=task_config)
    results = followup_generator.get_results(cur_query,
                                             embedding_key=embedding_key,
                                             num=num,
                                             formatting_function=formatting_function)
    print(results)
    results = formatting_answer_out(results)
    return results
