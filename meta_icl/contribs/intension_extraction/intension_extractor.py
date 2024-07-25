from meta_icl.core.online_icl.icl import EmbeddingICL
from meta_icl.core.utils import get_single_embedding, organize_text_4_embedding
from meta_icl.core.utils import message_formatting, call_llm_with_message
from meta_icl.contribs.intension_extraction.prompt.prompt_4_intension_extraction import formatting_intention_classification
import json



class IntentionAnalysis(EmbeddingICL):
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




def get_intention_analysis_results(cur_query: dict,
                                   embedding_key: list,
                                   base_model: str,
                                   embedding_pth,
                                   examples_pth,
                                   embedding_model=None,
                                   num=3):
    """

    :param cur_query: dict, {"chat_history": [{
      "用户": "优惠券用不了？",
      "客服": "让我查一下您的优惠券信息。"
    }],
    "user_query": "我的优惠券是NO123456789"}
    :param embedding_key: the key to request the embedding in cur_query.
    :param base_model: the model to generate the intention analysis results
    :param embedding_pth: the path storing the embedding vectors of the examples
    :param examples_pth: the path of the examples
    :param embedding_model: the model to get the embedding.
        currently only dashscope embedding model is available: "text_embedding_v1"
    :param num: the number of demonstration.md examples.
    :return: dict, example:{"user_intention": "无法观看已购买的课程，寻求帮助。", "intention_class": "无法观看课程" }
    """
    intention_analyzer = IntentionAnalysis(base_model=base_model,
                                           embedding_pth=embedding_pth,
                                           examples_pth=examples_pth,
                                           embedding_model=embedding_model)
    results = intention_analyzer.get_results(cur_query,
                                             embedding_key=embedding_key,
                                             num=num,
                                             formatting_function=formatting_intention_classification)
    print(results)
    results = json.loads(results)
    return results
