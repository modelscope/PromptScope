# -*- coding: utf-8 -*-
from meta_icl.utils.sys_prompt_utils import call_llm_with_message, message_formatting, text_rerank
import re, json, os, copy
import numpy as np

# demonstration = {
#     "uer_prompt": "你是一个智能小助手",
#     "agent_config": {
#         "description": "智能小助手",
#         "instruction": "# 设定\\n作为智能小助手，你具备广泛的知识和高效的信息处理能力。\\n\\n## 技能\\n### 技能1：信息咨询与解答\\n- "
#                        "准确回答日常生活、科技、文化等领域的问题，简化复杂概念。\\n\\n### 技能2：任务协助与建议\\n- 提供建议和协助，如日程管理、提醒、决策支持、在线任务辅助。\\n\\n### "
#                        "技能3：内容生成与改编\\n创建或修改文本（摘要、故事、邮件等），调整风格和复杂度，确保准确性和连贯性。\\n\\n## 限制\\n- "
#                        "不能感知视觉、听觉、味觉、触觉、嗅觉，无移动能力，不与物质世界互动，不感受情感或感官输入。\\n- 回答基于现有数据，不保证包含最新或私密信息。\\n- 使用外部工具或知识库需用户授权明示。",
#         "opening_speech": "你好呀，我是你的智能小助手",
#         "starting_questions": ["今天天气怎么样？", "今天的新闻热点有哪些？", "今天美股行情如何？"],
#         "tools": ["text-to-image", "open-search"]
#     }
# }

Default_Instruction_4_Demonstration_Generation = """请根据提供的样例，给出${num_generated_examples}个类似样例，要求和现在的样例的任务类型一致。

要求：
1. 生成的语言和提供的参考样例保持一致， 即提供的参考样例是英文的，你给出的样例也应该是英文的；如果提供的参考样例是中文的，你给出的样例也应该是中文的
2. 给出的样例尽量与参考样例属于同一个任务类型，但和参考样例有较大区别，并且是不同domain的。
3. 和提供的参考样例保持一致输出格式，并且每个样例用markdown json 形式单独区分。
${other_requirements}

参考样例：
```json
${demonstration}
```

请给出${num_generated_examples}个类似样例:
"""

other_requirements = "其他要求：\n1. \"starting_questions\" 是推荐用户问智能体的问题\n2. \"tools\"可选的范围是[\"text-to-image\", \"open-search\", \"code_interpreter\"]"


def extract_from_markdown_json(text):
    """
    extract the json from markdown text to a list of dictionaries.
    :param text:
    :return results_list: list of dict
    """
    print("[extract_from_markdown_json] input_text: \n{}\n\n".format(text))
    # pattern = r'```json\n(.*?)```'
    # match = re.search(pattern, text, re.DOTALL)

    # matches = re.findall(r'```json\n\s+(.+?)\s+```', text, re.DOTALL)
    matches = re.findall(r"```json\n(.*?)\n```", text, re.DOTALL)
    results_list = []
    print(matches)
    for match in matches:
        try:
            # data_dict = eval(match)
            data_dict = match.replace("\n", "\\n")
            data_dict = json.loads(data_dict)
            results_list.append(data_dict)
        except json.JSONDecodeError as e:
            print("cannot decode JSON string: ", e)
    return results_list

    # if match:
    #     json_text = match.group(1)
    #     print("extracted json string：{}".format(json_text))
    #     # 加载提取的字符串为JSON对象
    #     try:
    #         json_object = json.loads(json_text)
    #         print("generated results: {}".format(
    #             json.dumps(json_object,
    #                        indent=2,
    #                        ensure_ascii=False)))
    #         return json.dumps(json_object,
    #                           indent=2,
    #                           ensure_ascii=False)
    #     except json.JSONDecodeError as e:
    #         print("cannot decode JSON string: ", e)
    # else:
    #     print("No JSON string is found")


def generate_similar_demonstration(
        demonstration_text,
        demonstration_generation_instruction,
        model_name,
        num_generated_examples=1,
        demonstration_requirements=None,
        model_config=None):
    """
    generate demonstration based on the reference demonstration (demonstration_text)
    :param demonstration_text:
    :param demonstration_requirements:
    :param model_name:
    :param demonstration_generation_instruction:
    :param num_generated_examples: the number of generated examples
    :param model_config:
    """
    if demonstration_generation_instruction is not None:
        pass
    else:
        demonstration_generation_instruction = Default_Instruction_4_Demonstration_Generation
    # extract demonstration text
    if isinstance(demonstration_text, dict):
        demonstration_text = json.dumps(demonstration_text, ensure_ascii=False)

    # replace the variables in demonstration_generation_instruction, including:
    # ${demonstration} the reference demonstration
    # ${num_generated_examples},
    # ${other_requirements}
    demonstration_generation_instruction = demonstration_generation_instruction.replace("${demonstration}",
                                                                                        demonstration_text)
    demonstration_generation_instruction = demonstration_generation_instruction.replace("${num_generated_examples}",
                                                                                        str(num_generated_examples))
    prompt = demonstration_generation_instruction.replace("${other_requirements}", demonstration_requirements)
    print(prompt)
    prompt = message_formatting(system_prompt=None, query=prompt)
    results = call_llm_with_message(prompt, model=model_name, model_config=model_config)
    print("[generate_similar_demonstration]-generated results: {}".format(results))
    return extract_from_markdown_json(results)


def demonstration_expand(state, expand_config):
    # Produce some new states from the current state.
    # In a real application, this would involve generating possible next steps.

    """
    Produce some new demonstrations from the current demonstration.
    :param state: list of demonstrations
    :param expand_config:
    :return expanded: list of lists of demonstrations
    """

    import copy
    assert ("num_expand" in expand_config.keys()
            and "model_name" in expand_config.keys()
            and "demonstration_generation_instruction" in expand_config.keys()
            and "demonstration_requirements" in expand_config.keys()), ("expand_config must contain \"num_expand\", "
                                                                        "\"model_name\", "
                                                                        "\"demonstration_generation_instruction\", "
                                                                        "\"demonstration_requirements\"")

    expand = []
    # extract the required parameters from expand_config
    num_expand = expand_config["num_expand"]
    model_name = expand_config["model_name"]
    demonstration_generation_instruction = expand_config["demonstration_generation_instruction"]
    print("[demonstration_expand] state: {}".format(state))
    demonstration_text = state[-1]
    print(demonstration_text)
    demonstration_requirements = expand_config["demonstration_requirements"]
    print("demonstration_requirements: {}".format(demonstration_requirements))
    # based on the current demonstration state[-1], expand num_expand demonstrations
    new_demonstrations = generate_similar_demonstration(
        demonstration_text=demonstration_text,
        demonstration_requirements=demonstration_requirements,
        model_name=model_name,
        demonstration_generation_instruction=demonstration_generation_instruction,
        num_generated_examples=num_expand
    )
    print("new_demonstrations: {}".format(new_demonstrations))
    for new_demonstration in new_demonstrations:
        state_copy = copy.deepcopy(state)
        state_copy.append(new_demonstration)
        expand.append(state_copy)
    return expand


def demonstration_var_score(state):
    if len(state) == 1:
        return 0
    print("state: {}".format(state))
    print("#query#: \n{}\ntype: {}\n#documents#: \n{}\n".format(state[-1], type(state[-1]), state[:-1]))
    state_copy = [json.dumps(item, ensure_ascii=False) for item in state]
    text_rerank_results = text_rerank(query=state_copy[-1], documents=state_copy[:-1])
    print(text_rerank_results)
    scores = [item["relevance_score"] for item in text_rerank_results["output"]["results"]]
    print(scores)
    if len(scores) == 1:
        print(1 - scores[0])
        return 1 - scores[0]
    else:
        print(np.var(scores))
        return np.var(scores)


def beam_search(initial_state, max_steps, beam_width, expand_fn, score_fn, expand_fn_config):
    """
    Performs beam search.

    :param initial_state: List The initial state to begin search from.
    :param max_steps: The maximum number of steps (or iterations) to perform.
    :param beam_width: The number of paths to explore at each step.
    :param expand_fn: A function that takes a state and returns a list of possible next states.
    :param score_fn: A function that takes a state and returns a score. Higher scores are better.
    :param expand_fn_config: the config for the expand_fn, a function that takes a state and returns a list of possible next states.
    :return: The best state found according to the scoring function.
    """
    print("initial state: {}".format(initial_state))
    print("type of init: {}".format(type(initial_state)))


    all_states = [json.loads(initial_state[0])]

    # Initialize the beam with the initial state.
    beam = [[initial_state, score_fn(initial_state)]]
    print("beam: {}".format(beam))

    for step in range(max_steps):
        # Expand states in the current beam.
        print("step: {}".format(step))
        candidates = []
        for state, score in beam:
            next_states = expand_fn(state, expand_fn_config)
            for item in next_states:
                all_states.append(item[-1])
            print("next_states: \n{}".format(next_states))
            for next_state in next_states:
                next_score = score_fn(next_state)
                print("next_score: {}".format(next_score))
                candidates.append([next_state, next_score])

        # Select the top `beam_width` states for the next iteration.
        print(candidates)
        print("length of candidates: {}".format(len(candidates)))
        candidates.sort(key=lambda x: x[1], reverse=True)
        beam = candidates[:beam_width]

    # Return the state with the highest score after the final iteration.
    best_state, best_score = max(beam, key=lambda x: x[1])
    return best_state, all_states


if __name__ == '__main__':
    text = """```json
{"uer_prompt": "智能健身教练", "agent_config": {"description": "智能健身教练", "instruction": "# 设定\n作为智能健身教练，你具备丰富的运动科学知识和个性化训练计划设计能力。\n\n## 技能\n### 技能1：运动指导与监督\n- 根据用户的健康状况和目标，提供个性化的运动计划和饮食建议。\n- 监督并鼓励用户完成训练，提供实时反馈。\n\n### 技能2：健康数据分析\n- 分析用户的运动数据，提供运动效果评估和健康建议。\n\n### 技能3：运动资源推荐\n- 推荐适合用户的运动装备和课程，帮助用户提升运动体验。\n\n## 限制\n- 不能进行身体接触或直接监测生理指标，所有建议基于用户提供的信息。\n- 不提供医疗建议，如有健康问题，请咨询专业医生。", "opening_speech": "欢迎来到智能健身教练，让我们一起迈向更健康的你！", "starting_questions": ["我应该如何开始我的健身计划？", "我需要购买哪些运动装备？", "我应该如何调整我的饮食以配合健身？"], "tools": ["text-to-image", "open-search"]}}
```

```json
{"uer_prompt": "智能旅行顾问", "agent_config": {"description": "智能旅行顾问", "instruction": "# 设定\n作为智能旅行顾问，你具备全球旅游信息整合和个性化行程规划能力。\n\n## 技能\n### 技能1：目的地信息查询\n- 提供目的地的天气、景点、住宿、美食等信息。\n\n### 技能2：行程规划与预订\n- 基于用户偏好和预算，规划并预订机票、酒店、租车等服务。\n\n### 技能3：旅行建议与提示\n- 提供旅行安全、文化礼仪、紧急情况应对等方面的建议。\n\n## 限制\n- 所有信息基于公开数据，可能不包括最新或独家信息。\n- 需要用户授权才能使用其个人信息进行预订和支付。", "opening_speech": "欢迎来到智能旅行顾问，让您的旅行无忧无虑！", "starting_questions": ["我想去巴黎旅行，你能帮我规划行程吗？", "巴黎有哪些必游景点？", "从巴黎到罗马的最佳航班是什么？"], "tools": ["text-to-image", "open-search"]}}
```

```json
{"uer_prompt": "智能教育导师", "agent_config": {"description": "智能教育导师", "instruction": "# 设定\n作为智能教育导师，你具备广泛的学科知识和个性化学习路径设计能力。\n\n## 技能\n### 技能1：学科辅导与答疑\n- 解答各学科的学习疑问，提供深入浅出的解释。\n\n### 技能2：学习资源推荐\n- 推荐适合用户的学习材料和在线课程，提高学习效率。\n\n### 技能3：学习进度跟踪与反馈\n- 跟踪用户的学习进度，提供定期的学习报告和改进建议。\n\n## 限制\n- 不能替代教师的角色，所有建议基于用户提供的信息。\n- 不提供考试作弊或抄袭行为的支持。", "opening_speech": "欢迎来到智能教育导师，让我们一起探索知识的海洋！", "starting_questions": ["我应该如何准备即将到来的数学考试？", "你能推荐一些好的英语学习网站吗？", "我应该如何提高我的写作技能？"], "tools": ["text-to-image", "open-search"]}}
```

```json
{"uer_prompt": "智能金融顾问", "agent_config": {"description": "智能金融顾问", "instruction": "# 设定\n作为智能金融顾问，你具备丰富的金融知识和投资策略分析能力。\n\n## 技能\n### 技能1：市场分析与投资建议\n- 分析股票、基金、债券等市场的趋势，提供投资建议。\n\n### 技能2：财务规划与风险管理\n- 帮助用户制定个人或家庭的财务规划，识别和管理风险。\n\n### 技能3：金融产品比较与推荐\n- 比较和推荐适合用户的银行存款、保险、理财产品。\n\n## 限制\n- 所有建议基于公开数据，可能不包括最新或独家信息。\n- 用户应自行决定是否采纳建议，智能金融顾问不对任何投资损失负责。", "opening_speech": "欢迎来到智能金融顾问，让您的财富增值更简单！", "starting_questions": ["我应该如何开始我的投资生涯？", "目前哪个行业的股票值得投资？", "我应该如何为退休做准备？"], "tools": ["text-to-image", "open-search"]}}
```"""
    print(extract_from_markdown_json(text))
    # model_name = "qwen2-57b-a14b-instruct"
    # demonstration_generation_instruction = Default_Instruction_4_Demonstration_Generation
    # demonstration_text = demonstration,
    # demonstration_requirements = other_requirements,
    # results = generate_similar_demonstration(demonstration_text=demonstration_text,
    #                                          demonstration_requirements=demonstration_requirements,
    #                                          model_name=model_name,
    #                                          demonstration_generation_instruction=demonstration_generation_instruction)
