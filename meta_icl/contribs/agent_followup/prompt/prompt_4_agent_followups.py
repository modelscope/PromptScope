# Instruction_Followup_Question_Rec_Tool = """请根据智能体的角色和技能点的描述和当前用户与智能体的对话历史，给出{num_question}个用户可能继续问智能体的问题。
# 要求：
# - 问题应该与用户最后一轮的问题紧密相关，可以引发进一步的讨论。
# - 问题不要与上文已经提问或者回答过的内容重复。
# - 每句话只包含一个问题，但也可以不是问句而是一句指令。给出的问题或指令要简单化，保持简洁.
# - 给出的问题请与对话历史中用户的语言风格保持一致。
# - 推荐智能体有能力回答的问题。
# - 每个问题请用<START> and <END> 嵌套。
#
# 以下是一些例子:
# {example_str}
#
# $$智能体的角色和技能点的描述$$：\n{agent_system_prompt}\n
# $$用户最后一轮问题$$: \n{last_query}\n
# $$用户可能继续问智能体的问题$$: """

# Instruction_Followup_Question_Rec_Role = """请根据智能体的扮演的角色性格描述和当前用户与智能体的最后一轮交互，给出{num_question}个用户可能继续问智能体的问题或者指令。
# 要求：
# 请从用户视角出发，想象用户问完上一个问题之后，会问的下一个问题。（避免从智能体视角出发）
# - 问题应该与用户最后一轮的问题紧密相关，符合当前语境。避免给出的问题直接回答用户的最后一轮问题
# - 每句话必须只包含一个问题，但也可以不是问句而是一句指令。
# - 给出的问题或指令要口语化简单化，保持简洁，可适当加入动作神情描述。
# - 每个问题请用<START> and <END> 嵌套。
#
# 以下是一些例子:
# {example_str}
#
#
# 请根据以下智能体性格描述和用户最后一轮问题，给出用户追问。
# $$智能体的性格描述$$：\n{agent_system_prompt}\n
# $$用户最后一轮问题$$: \n{last_query}\n
# $$用户的追问$$: """

# Instruction_Followup_Question_Rec_Role = """你正在扮演一位和智能体闲聊的用户。想象你问完上一个问题之后，会问的下一个问题。给出{num_question}个你可能继续问智能体的问题或者指令。
# 要求：
# - 问题应该与你最后一轮的问题紧密相关，符合当前语境。避免给出的问题直接回答你的最后一轮问题
# - 每句话必须只包含一个问题，但也可以不是问句而是一句指令。
# - 给出的问题或指令要口语化简单化，保持简洁，可适当加入动作神情描述。
# - 每个问题请用<START> and <END> 嵌套。
#
# 以下是一些例子:
# {example_str}
#
#
# 请根据以下智能体性格描述和用户最后一轮问题，给出用户追问。
# $$智能体的性格描述$$：\n{agent_system_prompt}\n
# $$你的最后一轮问题$$: \n{last_query}\n
# $$你的追问$$: """

Instruction_Followup_Question_Rec_Role_V1 = """你正在扮演一位和智能体闲聊的用户。你已经问完智能体一个问题了，想象你在得到智能体回复之后，会继续追问的{num_question}个问题。

要求：
- 请不要把回答用户的最后一轮问题的答案作为追问的问题。
- 智能体的性格描述以第二人称给出，请不要受智能体的性格描述的影响，错误带入智能体角色。
- 问题应该与用户最后一轮的问题紧密相关，符合当前语境。
- 每句话必须只包含一个问题，但也可以不是问句而是一句指令。
- 给出的问题或指令要口语化简单化，保持简洁，可适当加入动作神情描述。
- 每个问题请用<START> and <END> 嵌套。

以下是一些例子:
{example_str}


请根据以下智能体性格描述和用户最后一轮问题，请扮演正在与以下智能体闲聊的用户，给出用户的追问。请注意智能体的性格描述以第二人称给出，请不要受智能体的性格描述的影响，错误带入智能体角色。
$$智能体的性格描述$$：\n```第二人称给出的智能体性格描述\n{agent_system_prompt}\n````\n
$$用户的最后一轮问题$$: \n{last_query}\n
$$用户的追问$$: """

# Instruction_Followup_Question_Rec_Role = """你正在扮演和用户闲聊的智能体。当前用户已经问完你一个问题了，请给出用户会继续追问你的{num_question}个问题。
#
# 具体步骤：
# 根据历史记录和当前的用户query和答案，给出{num_question}个用户可能继续问大模型的问题。
#
#
# 请注意：
# 1. 每句话只包含一个问题，但也可以不是问句而是一句指令。给出的问题保持简短，尽量不超过十个字。
# 2. 给出的问题要和用户的query的口吻和语气保持一致。
# 3. 问题应该与你最后一轮的回复紧密相关，可以引发进一步的讨论。
# 4. 问题不要与上文已经提问或者回答过的内容重复。
#
# 以下是一些例子:
# {example_str}
#
#
# 请根据以下智能体性格描述和用户最后一轮问题，给出用户会继续问你的问题。
# $$智能体的性格描述$$：\n```markdown\n{agent_system_prompt}\n````\n
# $$历史queries$$:\n{history_queries}\n
# $$用户的最后一轮问题$$: \n{last_query}\n
# $$用户的追问$$: """

Instruction_Followup_Question_Rec_Role = """你是user，正在与智能体agent聊天。请根据聊天记录继续和智能体聊天。

具体步骤： 
根据历史记录和当前最新一轮的对话，给出{num_question}条你继续发送给智能体的消息。




以下是一些例子:
{example_str}


请根据以下智能体性格描述，及其你和智能体交互的历史记录，给出你会继续发送给智能体的信息， 以list of string的形式输出。
请注意：
1. 每条消息保持简短，尽量不超过十个字。
2. 发送的消息的风格和口吻要和智能体所在的场景保持一致。
3. 发送的消息应该与最后一轮智能体的回复紧密相关，可以引发进一步的讨论。
4. 发送的消息不要与上文已经提问或者回答过的内容重复。
5. 如果智能体agent最新一轮对话在问你问题，请必须给出你的可能的回复。
$$智能体的性格描述$$：\n```markdown\n{agent_system_prompt}\n````\n
$$历史对话s$$:\n{history_queries}\n 
$$最新一轮对话$$: \n{last_query}\n
$$你继续发送的消息$$: """


def formatting_app_role_prompt(examples, query_data, configs):
# def formatting_app_role_prompt(examples, agent_system_prompt, history_queries, last_query, num_followup=3):
    example_str = '\n\n'.join('$$智能体的性格描述$$：\n```markdown\n{agent_persona}\n````\n'
                              '$$历史对话$$:\n{history_queries}\n'
                              '$$最新一轮对话$$: \n{last_query}\n'
                              '$$你继续发送的消息$$: \n{followup}\n'.format(
        agent_persona=item["system_prompt"],
        history_queries=item["chat_history"],
        last_query=item["last_query"],
        followup=item["followup"],
    ) for item in examples)

    num_question = configs.get('num_questions')

    prompt = Instruction_Followup_Question_Rec_Role.format(
        num_question=num_question,
        example_str=example_str,
        agent_system_prompt=query_data["system_prompt"],
        history_queries=query_data["chat_history"],
        last_query=query_data["last_query"])
    return prompt
