from meta_icl.icl.ICL import CustomizedICL
from meta_icl.utils.utils import get_single_embedding, organize_text_4_embedding
from meta_icl.utils.sys_prompt_utils import message_formatting, call_llm_with_message
from meta_icl.contribs.intension_extraction.prompt.prompt_4_intension_extraction import \
    formatting_intention_classification
import json


class AppMainFollowupStr(CustomizedICL):
    def __init__(self, base_model,
                 embedding_pth,
                 examples_pth,
                 embedding_model=None
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
                         examples_pth=examples_pth)
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
                         num=3):
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
    :param num: the number of demonstration examples.
    :return: list of str, list of followup questions.
    """
    followup_generator = AppMainFollowupStr(base_model=base_model,
                                            embedding_pth=embedding_pth,
                                            examples_pth=examples_pth,
                                            embedding_model=embedding_model)
    results = followup_generator.get_results(cur_query,
                                             embedding_key=embedding_key,
                                             num=num)
    print(results)
    results = json.loads(results)
    return results
