from meta_icl.utils.sys_prompt_utils import call_llm_with_message

Followup_Question_Rec = """任务描述：请根据大模型和当前用户的对话历史，给出{num_question}个用户可能继续问大模型的问题。\n

具体步骤： 
1. 请先给出当前用户的query的回答会包含的内容总，
2. 然后根据历史记录和当前的用户query和答案，给出{num_question}个用户可能继续问大模型的问题。

以下是示例： 
$$历史queries$$:\n["交社保最好是只在一个地方交吗，如果换了工作地，原工作地交的社保会如何", "医疗保险中断会如何"]\n 
$$当前query$$: "最重要的是养老和医疗保险吗，其中养老保险最好不中断缴纳是吗" 
$$用户可能继续问智能体的问题$$: {"reasons": "当前的query的回答会包括养老和医疗保险的重要性，和养老保险中断缴纳带来的后果。",
"followup_questions": ["五险一金包括？", "养老保险转移后，缴费年限怎么算？","如何办理医保转移接续？"]}


$$历史queries$$:\n{history_queries}\n 
$$当前query$$: {cur_query}
$$用户可能继续问智能体的问题$$: 
"""

