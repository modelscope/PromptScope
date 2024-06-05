from meta_icl.utils.sys_prompt_utils import load_csv, sav_json
from meta_icl.utils.utils import get_current_date
import pandas as pd
import os




if __name__ == '__main__':

    csv_pth = "data/data_build_stocks/others/0604.csv"
    json_sav_dir = "data/data_build_stocks/others/"
    group_by_filed = "session_id"

    combine_session(csv_pth, json_sav_dir, group_by_filed, selection_filed=None)

    """
    
    data_pth = "data/data_build_stocks/用户历史对话记录.csv"
    data = pd.read_csv(data_pth)
    print(data.keys())
    sav_dir = "data/data_build_stocks"

    grouped = data.groupby("session_id")
    sessions = []
    for session_id, group in grouped:
        tmp = group.to_dict(orient='records')
        conversations = []
        print(tmp)
        print(len(tmp))

        # break
        for item in tmp:
            conversations.append({
                "query": item["query"],
                "answer": item["answer"]
            })
        session_dict = {
            'session_id': session_id,
            'conversations': conversations  # List of rows for this session
        }

        sessions.append(session_dict)
    sav_json(sessions, os.path.join(sav_dir, "organize_conversation_history.json"))
    """
