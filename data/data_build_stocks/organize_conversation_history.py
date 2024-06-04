from meta_icl.utils.sys_prompt_utils import load_csv

if __name__ == '__main__':
    data_pth = "data/data_build_stocks/用户历史对话记录.csv"
    data = load_csv(data_pth)
    print(data.keys())

    json_data = []
    cur_session_id = ""

    for idx in range(len(data["\ufeffsession_id"])):
        if idx != cur_session_id:
            cur_session_id = idx
            tmp = {"history_queries": [], "cur_query": ""}

