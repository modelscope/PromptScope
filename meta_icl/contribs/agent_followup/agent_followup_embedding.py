from meta_icl.core.online_icl.icl import EmbeddingICL
from meta_icl.core.utils import timer

from meta_icl.contribs.agent_followup.prompt.prompt_4_agent_followups import formatting_answer_out


class AgentFollowupEmbedding(EmbeddingICL):
    def __init__(self, embedding_icl_configs, task_configs):
        base_model = task_configs.get("base_model")
        embedding_pth = embedding_icl_configs.get("embedding_pth")
        examples_pth = embedding_icl_configs.get("examples_pth")
        embedding_model = embedding_icl_configs.get("embedding_model")
        self.task_configs = task_configs
        retriever_key_list = embedding_icl_configs.get("search_key")

        super().__init__(base_model=base_model,
                         embedding_pth=embedding_pth,
                         examples_pth=examples_pth,
                         retriever_key_list=retriever_key_list,
                         task_configs=task_configs,
                         embedding_model=embedding_model)


@timer
def get_agent_embedding_followup_results(cur_query: dict,
                                         task_configs: dict,
                                         icl_configs: dict,
                                         formatting_function,
                                         **kwargs):
    """
    """
    # formatting_function = kwargs.get("formatting_function")
    embedding_retriever_configs = icl_configs.get("embedding_retriever_configs")
    followup_generator = AgentFollowupEmbedding(embedding_icl_configs=embedding_retriever_configs,
                                                task_configs=task_configs)
    num_selection = embedding_retriever_configs.get("topk", 3)
    results = followup_generator.get_results(
        cur_query,
        formatting_function=formatting_function,
        num=num_selection, **kwargs
    )
    print(results)
    results = formatting_answer_out(results)
    return results
