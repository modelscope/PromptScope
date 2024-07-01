from meta_icl.contribs.app_main_followup.app_main_followup_str import get_followup_results
from meta_icl.core.utils import load_json_file

if __name__ == '__main__':
    config_pth = "conf/app_followup_configs/app_followup_str_conf.json"
    conf = load_json_file(config_pth)
    cur_query = {
        "chat_history": [
            "交社保最好是只在一个地方交吗，如果换了工作地，原工作地交的社保会如何",
            "医疗保险中断会如何"
        ],
        "last_query": "最重要的是养老和医疗保险吗，其中养老保险最好不中断缴纳是吗",
    }

    results = get_followup_results(cur_query, embedding_key=conf["icl_configs"]["embedding_key"],
                                   base_model=conf["task_configs"]["base_model"],
                                   embedding_pth=conf["icl_configs"]["embedding_pth"],
                                   examples_pth=conf["icl_configs"]["examples_pth"],
                                   embedding_model=conf["icl_configs"]["embedding_model"],
                                   model_config=None,
                                   task_config=conf["task_configs"],
                                   num=conf["icl_configs"]["topk"], )
    print(results)
