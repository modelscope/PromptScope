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


if __name__ == '__main__':
    data_pth = "data/app_data/agent_followup_data/demonstrations/role_followup.json"
    data = load_json_file(data_pth)
    reformat_demos = []
    for item in data:
        tmp = {
            "system_prompt": item["system_prompt"],
            "chat_history": [],
            "last_query": {
                "user": item["last_query"],
                "agent": item["last_res"]
            },
            "followup": extract_feedbacks(item["followup"])
        }
        reformat_demos.append(tmp)

    exist_pth = "data/app_data/agent_followup_data/demonstrations/role_followups_from_xingye_full.json"
    xinye_data = load_json_file(exist_pth)
    reformat_demos.extend(xinye_data)

    sav_json(reformat_demos, "data/app_data/agent_followup_data/demonstrations/role_followups_full_reformat.json")
