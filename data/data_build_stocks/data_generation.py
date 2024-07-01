from meta_icl.core.utils import call_llm_with_message

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

if __name__ == '__main__':
    from meta_icl.core.utils import message_formatting
    history_queries = []
    cur_query = """总结一下要点：我们的产品具有以下特点：
保障全面：我们的车险产品涵盖了车辆损失险、第三者责任险、车上人员责任险、盗抢险、玻璃单独破碎险、自燃损失险、车身划痕险、不计免赔险等多种险种，可以为您的车辆提供全方位的保障。
理赔快捷：我们拥有专业的理赔团队和先进的理赔系统，可以为您提供快速、便捷的理赔服务。您只需拨打我们的客服热线，我们将在第一时间为您处理理赔事宜。
价格合理：我们的车险产品价格合理，可以根据您的车辆情况和驾驶记录进行个性化定价，让您享受到最优惠的价格。
服务周到：我们为客户提供全方位的服务，包括在线投保、保单查询、理赔服务、道路救援等，让您享受到便捷、高效的服务。
如果您对我们的产品感兴趣，请随时联系我们，我们将竭诚为您服务。"""
    query = Followup_Question_Rec_Fix_Example.format(num_question=3,
                                                     history_queries=history_queries,
                                                     cur_query=cur_query)
    messages = message_formatting(system_prompt=None, query=query)
    model = "GPT4"
    results = call_llm_with_message(messages=messages, model=model)
    print(results)





