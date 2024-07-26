# from http import HTTPStatus
# import dashscope
# from meta_icl.core.utils import *
#
#
# def simple_multimodal_conversation_call():
#     """Simple single round multimodal conversation call.
#     """
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"image": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"},
#                 {"text": "这是什么?"}
#             ]
#         }
#     ]
#     response = dashscope.MultiModalConversation.call(model='qwen-vl-plus',
#                                                      messages=messages)
#     # The response status_code is HTTPStatus.OK indicate success,
#     # otherwise indicate request is failed, you can get error code
#     # and message from code and message.
#     if response.status_code == HTTPStatus.OK:
#         print(response)
#     else:
#         print(response.code)  # The error code.
#         print(response.message)  # The error message.
#
#
# if __name__ == '__main__':
#     simple_multimodal_conversation_call()
#
#
# import json
#
# json_text = '''
# {
#   "system_prompt": "你是一只宠物猫，名字叫毛球，你的主人是个大学生，最近他因为考试压力大，总是忘记按时给你喂食。",
#   "chat_history": [
#     {
#       "role": "user",
#       "content": "（你饿得喵喵叫）"
#     },
#     {
#       "role": "assistant",
#       "content": "（主人抬起头，看着你，一脸疲惫）哎呀，毛球，对不起，我忘了，马上给你弄吃的。"
#     }
#   ],
#   "followup": [
#     "（你欢快地跑向猫碗）",
#     "（主人准备猫粮，你耐心等待）",
#     "（主人一边喂食，一边轻声道歉）"
#   ]
# }
# '''
#
# try:
#     data_dict = json.loads(json_text)
#     print(data_dict)
# except json.JSONDecodeError as e:
#     print(f"Error decoding JSON: {e}")
from meta_icl.core.utils.utils import load_file, get_current_date, sav_csv
import copy
from meta_icl.core.utils.sys_prompt_utils import message_formatting, call_llm_with_message, load_csv
from meta_icl.core.utils.evaluator_utls import single_round_qa_judge
from loguru import logger
import re

logger.add(f"logs/test_{get_current_date()}.log", rotation="1 MB")

def extract_rating(text):
    result = re.search(r'\[\[(.*?)\]\]', text)
    if result:
        print(result.group(1))
        return result.group(1)
    else:
        print("没有找到匹配的内容")
        return ""
def judge_results(csv_pth, query_col_name, ans_col_name_list, model_name):
    data = load_file(csv_pth)
    results = copy.deepcopy(data)
    print(data.keys())
    sav_pth = csv_pth.replace(".csv", "_with_rating.csv")
    total_data = len(data["en_example"])
    for res_key_name in ans_col_name_list:
        results[f"{res_key_name}_rating"] = [1]*total_data
        results[f"{res_key_name}_rating_score"]= [1]*total_data
    for idx in range(total_data):
        query = data[query_col_name][idx]
        for res_key_name in ans_col_name_list:
            ans = data[res_key_name][idx]
            try:
                rating = single_round_qa_judge(query, ans, model_name)
            except:
                rating = ''
            results[f"{res_key_name}_rating"][idx] = rating
            logger.info(f"idx: {idx}, res_key_name: {res_key_name}, res: {rating}")
            try:
                rating_score = extract_rating(rating)
            except:
                rating_score = ''
            results[f"{res_key_name}_rating_score"][idx] = rating_score
            logger.info(f"idx: {idx}, res_key_name: {res_key_name}, rating_score: {rating_score}")
    sav_csv(results, sav_pth)

    import pandas as pd
    results_pd = pd.DataFrame(results)
    xlsx_pth = sav_pth.replace(".csv", ".xlsx")
    results_pd.to_excel(xlsx_pth)






if __name__ == '__main__':
    # # data_pth = "data/prompt_data/百炼国际站模版 - 副本_en-US_v1.csv"
    # # sav_pth = f"data/prompt_data/百炼国际站模版 - 副本_en-US_v1_run_date_{get_current_date()}.json"
    # # data = load_file(data_pth)
    # # results = copy.deepcopy(data)
    # # print(data.keys())
    # # total_data = len(data["en_example"])
    # # model_name_2_call = ["Qwen-plus", "Qwen-turbo", "Qwen-max"]
    # #
    # # for model_name in model_name_2_call:
    # #     key_name = f"{model_name}_results"
    # #     results[key_name] = []
    # #     for idx in range(total_data):
    # #         query = data["en_example"][idx]
    # #         message = message_formatting(query=query, history=None)
    # #         try:
    # #             res = call_llm_with_message(messages=message, model=model_name)
    # #             results[f"{model_name}_results"].append(res)
    # #             logger.info(f"idx: {idx}, model: {model_name}, res: {res}")
    # #         except:
    # #             logger.error("call_failed.")
    # #             results[f"{model_name}_results"].append("")
    # #     sav_csv(results, sav_pth)
    csv_pth = "data/prompt_data/百炼国际站模版 - 副本_en-US_v1_run_date_2024-07-25 16:27:26.csv"
    data = load_csv("data/prompt_data/百炼国际站模版 - 副本_en-US_v1_run_date_2024-07-25 16:27:26.csv")
    print(data.keys())

    query_col_name = 'en_example'
    ans_col_name_list = ['Qwen-plus_results', 'Qwen-turbo_results', 'Qwen-max_results']
    model_name = "gpt4"
    judge_results(csv_pth, query_col_name, ans_col_name_list, model_name)


    # pth = "data/prompt_data/百炼国际站模版 - 副本_en-US_v1_run_date_2024-07-25 16:27:26_with_rating.csv"
    # sav_pth = pth.replace(".csv", ".xlsx")
    # data = load_csv(pth)
    # ans_col_name_list = ['Qwen-plus_results', 'Qwen-turbo_results', 'Qwen-max_results']
    #
    #
    #
    #
    #
    #
    # import pandas as pd
    # data = pd.DataFrame(data)
    # for key in ans_col_name_list:
    #     col_name = f"{key}_rating_score"
    #     data[col_name] = data[col_name].apply(lambda x: extract_rating(x))
    # data.to_excel(sav_pth, index=False)










