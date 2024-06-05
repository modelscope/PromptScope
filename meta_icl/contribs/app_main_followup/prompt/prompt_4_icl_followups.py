Followup_Question_Rec_Fix_Example = """任务描述：请根据大模型和当前用户的对话历史，给出{num_question}个用户可能继续问大模型的问题。\n

具体步骤： 
1. 请先给出当前用户的query的回答会包含的内容总，
2. 然后根据历史记录和当前的用户query和答案，给出{num_question}个用户可能继续问大模型的问题。

请注意：
1. 每句话只包含一个问题，但也可以不是问句而是一句指令。给出的问题保持简短，尽量不超过十个字。
2. 给出的问题要和用户的query的口吻和语气保持一致。
3. 问题应该与你最后一轮的回复紧密相关，可以引发进一步的讨论。
4. 问题不要与上文已经提问或者回答过的内容重复。
5. 推荐你有能力回答的问题。

以下是示例： 
$$历史queries$$:\n["交社保最好是只在一个地方交吗，如果换了工作地，原工作地交的社保会如何", "医疗保险中断会如何"]\n 
$$当前query$$: "最重要的是养老和医疗保险吗，其中养老保险最好不中断缴纳是吗" 
$$用户可能继续问智能体的问题$$: {{"reasons": "当前的query的回答会包括养老和医疗保险的重要性，和养老保险中断缴纳带来的后果。",
"followup_questions": ["五险一金包括？", "养老保险转移后，缴费年限怎么算？","如何办理医保转移接续？"]}}


$$历史queries$$:\n{history_queries}\n 
$$当前query$$: {cur_query}
$$用户可能继续问智能体的问题$$: 
"""

Followup_Question_Rec_Str_icl = """任务描述：请根据大模型和当前用户的对话历史，给出{num_question}个用户可能继续问大模型的问题。\n

具体步骤： 
根据历史记录和当前的用户query和答案，给出{num_question}个用户可能继续问大模型的问题。

请注意：
1. 每句话只包含一个问题，但也可以不是问句而是一句指令。给出的问题保持简短，尽量不超过十个字。
2. 给出的问题要和用户的query的口吻和语气保持一致。
3. 问题应该与你最后一轮的回复紧密相关，可以引发进一步的讨论。
4. 问题不要与上文已经提问或者回答过的内容重复。

以下是示例： 
{demonstration_examples}


$$历史queries$$:\n{history_queries}\n 
$$当前query$$: {cur_query}
$$用户可能继续问智能体的问题$$: 
"""

Followup_Question_Rec_M_icl = """任务描述：请根据大模型和当前用户的对话历史，给出{num_question}个用户可能继续问大模型的问题。\n

具体步骤： 
根据历史记录和当前的用户query和答案，给出{num_question}个用户可能继续问大模型的问题。

请注意：
1. 每句话只包含一个问题，但也可以不是问句而是一句指令。给出的问题保持简短，尽量不超过十个字。
2. 给出的问题要和用户的query的口吻和语气保持一致。
3. 问题应该与你最后一轮的回复紧密相关，可以引发进一步的讨论。
4. 问题不要与上文已经提问或者回答过的内容重复。

以下是示例： 
{demonstration_examples}


$$历史queries$$:\n{history_queries}\n 
$$当前query$$: {cur_query}
$$当前回复$$: {cur_answer}
$$用户可能继续问智能体的问题$$: 
"""

def formatting_followups_in_prompt(question_list):
    return question_list
    # return "\n"+ "\n".join(question for question in question_list)
def formatting_answer_out(text):
    question_list = eval(text)
    if isinstance(question_list, list):
        return question_list
    else:
        print(f"Failed to generate questions: {text}")
    return question_list

def formatting_str_type_main_chat(examples, query_data, configs):
    example_str = "\n".join(f'$$历史queries$$:\n{item["chat_history"]}\n'
                            f'$$当前query$$: {item["last_query"]}\n'
                            # f'$$用户可能继续问智能体的问题$$: {{\"followup_questions\": \"{item["followup_questions"]}\"}}'
                            f'$$用户可能继续问智能体的问题$$: {formatting_followups_in_prompt(item["followup_questions"])}'
                            for item in examples)
    num_question = configs["num_questions"]
    query = Followup_Question_Rec_Str_icl.format(num_question=num_question,
                                                 demonstration_examples=example_str,
                                                 cur_query=query_data["last_query"],
                                                 history_queries=query_data["chat_history"])
    return query


def formatting_multimodal_type_main_chat(examples, query_data, configs):
    example_str = "\n".join(f'$$历史queries$$:\n{item["chat_history"]}\n'
                            f'$$当前query$$: {item["last_query"]}\n'
                            f'$$当前回复$$: {item["last_answer"]}\n'
                            f'$$用户可能继续问智能体的问题$$: {formatting_followups_in_prompt(item["followup_questions"])}'
                            for item in examples)
    num_question = configs["num_questions"]
    query = Followup_Question_Rec_Str_icl.format(num_question=num_question,
                                                 demonstration_examples=example_str,
                                                 cur_query=query_data["last_query"],
                                                 history_queries=query_data["chat_history"],
                                                 cur_answer=query_data["last_answer"])
    return query
