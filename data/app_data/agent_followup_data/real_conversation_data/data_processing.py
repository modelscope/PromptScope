from meta_icl.core.utils.utils import combine_session, convert_xlsx_2_json
# from meta_icl.core.utils.sys_prompt_utils import


if __name__ == '__main__':


    # convert_xlsx_2_json(
    #     json_file_path='data/app_data/agent_followup_data/real_conversation_data/智能体追问/role_conversation.json',
    #     excel_file_path='data/app_data/agent_followup_data/real_conversation_data/智能体追问/role_conversation.xlsx',
    #     eval_key_list=[]
    # )
    # combine_session(
    #     csv_pth='data/app_data/agent_followup_data/real_conversation_data/智能体追问/role_conversation.csv',
    #     json_sav_dir='data/app_data/agent_followup_data/real_conversation_data/智能体追问',
    #     group_by_filed='session_id',
    #     selection_filed=["name",
    #                      "instructions",
    #                      "query",
    #                      "answer",
    #                      "log_time"],
    #     prefix='session_role_conversation',
    #     # mapping_filed={'question': 'question', 'answer': 'answer'}
    # )
    # combine_session(
    #     csv_pth='data/app_data/agent_followup_data/real_conversation_data/智能体追问/special_case_哄女朋友.csv',
    #     json_sav_dir='data/app_data/agent_followup_data/real_conversation_data/智能体追问',
    #     group_by_filed='session_id',
    #     selection_filed=["name",
    #                      "instructions",
    #                      "query",
    #                      "answer",
    #                      "log_time"],
    #     prefix='session_role_conversation_gf',
    #     # mapping_filed={'question': 'question', 'answer': 'answer'}
    # )
    combine_session(
        csv_pth='data/app_data/agent_followup_data/real_conversation_data/智能体追问/tool_conversation.csv',
        json_sav_dir='data/app_data/agent_followup_data/real_conversation_data/智能体追问',
        group_by_filed='session_id',
        selection_filed=["name",
                         "instructions",
                         "query",
                         "answer",
                         "log_time"],
        prefix='session_tool_conversation',
        # mapping_filed={'question': 'question', 'answer': 'answer'}
    )