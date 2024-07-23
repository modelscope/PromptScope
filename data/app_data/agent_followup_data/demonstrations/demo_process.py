from meta_icl.core.utils.sys_prompt_utils import load_json_file, sav_json
from typing import List
import re


def extract_feedbacks(text: str) -> List[str]:
    """
    using regular express match the content between <START> and <END>
    :param text: str
    :return: list of str
    """
    # using regular express match the content between <START> and <END>
    matches = re.findall(r'<START>(.*?)<END>', text, re.DOTALL)

    # put each content into a list
    suggestions = [match.strip() for match in matches]
    return suggestions

def del_last_answer(json_file_path: str, sav_pth=None):
    """
    delete the last answer in the json file
    :param json_file_path: str
    :return: None
    """
    data = load_json_file(json_file_path)
    for idx in range(len(data)):
        data[idx]["last_query"] = data[idx]["last_query"]["user"]

    if sav_pth is not None:
        pass
    else:
        sav_pth = json_file_path.replace(".json", "_del_last_ans.json")
        sav_json(data, sav_pth)



if __name__ == '__main__':
    json_pth_list = ["data/app_data/agent_followup_data/demonstrations/role_followups_full_reformat.json",
                     "data/app_data/agent_followup_data/demonstrations/tool_followups_reformat.json"]
    for json_pth in json_pth_list:
        del_last_answer(json_pth)


    # # data_pth = "data/app_data/agent_followup_data/demonstrations/role_followup.json"
    # data_pth = "data/app_data/agent_followup_data/demonstrations/tool_followup.json"
    # json_sav_pth = "data/app_data/agent_followup_data/demonstrations/tool_followups_reformat.json"
    # data = load_json_file(data_pth)
    # reformat_demos = []
    # for item in data:
    #     tmp = {
    #         "system_prompt": item["system_prompt"],
    #         "chat_history": [],
    #         "last_query": {
    #             "user": item["last_query"],
    #             "agent": item["last_res"]
    #         },
    #         "followup": extract_feedbacks(item["followup"])
    #     }
    #     reformat_demos.append(tmp)
    #
    # # exist_pth = "data/app_data/agent_followup_data/demonstrations/role_followups_from_xingye_full.json"
    # # xinye_data = load_json_file(exist_pth)
    # # reformat_demos.extend(xinye_data)

    # sav_json(reformat_demos, json_sav_pth)