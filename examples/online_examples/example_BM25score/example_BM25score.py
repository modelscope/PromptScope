from meta_icl.contribs.agent_followup.agent_followup_bm25 import get_BM25_followup_results
from meta_icl.core.utils.config_utils import load_config

if __name__ == '__main__':
    # BM25_pth = "data/icl_bm25_demo/animal_index_bm25"
    #
    #
    #
    # retriever = BM25Retriever(bm25_index_pth=BM25_pth)
    # print(retriever.topk_selection("a cat", 3))

    config_pth = "conf/app_followup_configs/online_icl_config/online_icl_config_bm25.json"
    conf = load_config(config_pth)
    cur_query = {
        "chat_history": [
            "交社保最好是只在一个地方交吗，如果换了工作地，原工作地交的社保会如何",
            "医疗保险中断会如何"
        ],
        "last_query": "最重要的是养老和医疗保险吗，其中养老保险最好不中断缴纳是吗",
    }

    results = get_BM25_followup_results(cur_query=cur_query,
                                        task_configs=conf["task_configs"],
                                        icl_configs=conf["icl_configs"],
                                        file_type="no")
    print(results)
