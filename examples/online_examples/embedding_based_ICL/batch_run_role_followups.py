import os.path

from meta_icl.contribs.agent_followup.agent_followup_embedding import get_agent_embedding_followup_results
from meta_icl.core.utils.config_utils import load_config
from meta_icl.contribs.agent_followup.prompt.prompt_4_agent_followups import formatting_app_role_prompt
from meta_icl.core.utils.sys_prompt_utils import load_json_file
import copy
from meta_icl.core.utils import add_duration, get_current_date, sav_json
from loguru import logger
import argparse
parser = argparse.ArgumentParser(description='Agent followups')

# 添加位置参数
parser.add_argument("--online_config_pth", type=str,
                    help="the config of online config", default="")
parser.add_argument("--data_pth", type=str,
                    help="the path of the data", default="")
parser.add_argument("--sav_dir", type=str,
                    help="the dir to save results", default="")
parser.add_argument("--prefix", type=str,
                    help="the dir to save results", default="")
parser.add_argument("--query_only", type=int,
                    help="1: if only use query, else use query + answer", default=1)

def from_cov_2_cur_data(cov_data: list, max_history: int = 5, query_only=False):
    chat_history = []
    cur_query_list = []

    for i in range(len(cov_data)):
        last_query = {
            "user": cov_data[i]['query'],
            "agent": cov_data[i]['answer']
        }
        system_prompt = cov_data[i]['instructions']
        chat_history_i = copy.deepcopy(chat_history)
        cur_query = {
            "system_prompt": system_prompt,
            "chat_history": chat_history_i,
            "last_query": last_query if not query_only else last_query["user"]
        }

        cur_query_list.append(cur_query)
        if len(cur_query_list) > max_history:
            cur_query_list = cur_query_list[-5:]

    return cur_query_list


@add_duration
def get_results_with_duration(cur_query, conf, formatting_function):
    results = get_agent_embedding_followup_results(cur_query,
                                                   task_configs=conf.get("task_configs"),
                                                   icl_configs=conf.get("icl_configs"),
                                                   formatting_function=formatting_function)
    return results


if __name__ == '__main__':


    # config_pth = "conf/agent_followup_configs/online_icl_config/online_icl_config1.yaml"
    # conf = load_config(config_pth)
    #
    # # data_pth = ("data/app_data/agent_followup_data/real_conversation_data/智能体追问/session_role_conversation_ver_2024-07"
    # #             "-10 11:58:00.json")
    # # data = load_json_file(data_pth)
    # # results_data = copy.deepcopy(data)
    # # cur_time = get_current_date()
    # # sav_dir = "data/app_data/agent_followup_data/real_conversation_data/智能体追问/results"
    # # prefix = "role_followups"
    # # base_model = conf.get("task_configs").get("base_model")
    # #
    # # config_pth = "conf/agent_followup_configs/online_icl_config/online_icl_config1.yaml"
    # # conf = load_config(config_pth)
    #
    # data_pth = ("data/app_data/agent_followup_data/real_conversation_data/智能体追问/session_role_conversation_gf_ver_2024-07-18 19:57:54.json")
    # data = load_json_file(data_pth)
    # results_data = copy.deepcopy(data)
    # cur_time = get_current_date()
    # sav_dir = "data/app_data/agent_followup_data/real_conversation_data/智能体追问/results"
    # prefix = "role_followups_gf"
    # base_model = conf.get("task_configs").get("base_model")

    args = parser.parse_args()
    config_pth = args.online_config_pth
    conf = load_config(config_pth)
    data_pth = args.data_pth
    sav_dir = args.sav_dir
    prefix = args.prefix
    base_model = conf.get("task_configs").get("base_model")
    data = load_json_file(data_pth)
    results_data = copy.deepcopy(data)
    cur_time = get_current_date()
    query_only = args.query_only == 1

    log_pth = os.path.join(sav_dir, f"{prefix}_{base_model}_followups_{cur_time}.log")
    logger.add(log_pth, rotation="5 MB")

    sav_pth = os.path.join(sav_dir, f"{prefix}_{base_model}_followups_{cur_time}.json")
    total = len(data)
    for session_idx in range(total):
        logger.info(f"processing session {session_idx}/{total}")
        session = data[session_idx]
        session_id = session.get("session_id")
        conversations = session.get("conversations")
        cur_data_list = from_cov_2_cur_data(conversations, query_only=query_only)
        for chat_id in range(len(cur_data_list)):
            cur_query = cur_data_list[chat_id]
            try:
                results, duration = get_results_with_duration(cur_query, conf=conf,
                                                              formatting_function=formatting_app_role_prompt)
                print(results)
                results_data[session_idx]['conversations'][chat_id]["followups"] = results
                results_data[session_idx]['conversations'][chat_id]["cost"] = duration
            except Exception as e:
                print(e)
                logger.error(e)

            sav_json(results_data, sav_pth)
