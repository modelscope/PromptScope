# -*- coding: utf-8 -*-
from meta_icl.utils.sys_prompt_utils import call_llm_with_message, message_formatting, text_rerank
import re, json, os

demonstration = """```json
{
  "uer_prompt": "你是一个智能小助手",
  "agent_config": {
    "description": "智能小助手",
    "instruction": "# 设定\\n作为智能小助手，你具备广泛的知识和高效的信息处理能力。\\n\\n## 技能\\n### 技能1：信息咨询与解答\\n- 准确回答日常生活、科技、文化等领域的问题，简化复杂概念。\\n\\n### 技能2：任务协助与建议\\n- 提供建议和协助，如日程管理、提醒、决策支持、在线任务辅助。\\n\\n### 技能3：内容生成与改编\\n创建或修改文本（摘要、故事、邮件等），调整风格和复杂度，确保准确性和连贯性。\\n\\n## 限制\\n- 不能感知视觉、听觉、味觉、触觉、嗅觉，无移动能力，不与物质世界互动，不感受情感或感官输入。\\n- 回答基于现有数据，不保证包含最新或私密信息。\\n- 使用外部工具或知识库需用户授权明示。",
    "opening_speech": "你好呀，我是你的智能小助手"
    "starting_questions": ["今天天气怎么样？", "今天的新闻热点有哪些？", "今天美股行情如何？"],
    "tools": ["text-to-image", "open-search"]
  }
}
```
"""

Default_Instruction_4_Demonstration_Generation = """请根据提供的样例，给出一个类似样例，要求和现在的样例的任务类型一致。

要求：
1. 生成的语言和提供的参考样例保持一致， 即提供的参考样例是英文的，你给出的样例也应该是英文的；如果提供的参考样例是中文的，你给出的样例也应该是中文的
2. 给出的样例尽量与参考样例属于同一个任务类型，但和参考样例有较大区别，并且是不同domain的。
3. 和提供的参考样例保持一致输出格式
{other_requirements}

参考样例：
{demonstration}

请给出你的样例:
"""

other_requirements = "其他要求：\n1. \"starting_questions\" 是推荐用户问智能体的问题\n2. \"tools\"可选的范围是[\"text-to-image\", \"open-search\", \"code_interpreter\"]"


def extract_from_markdown_json(text):
    print("input_text: {}".format(text))
    pattern = r'```json\n(.*?)```'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        json_text = match.group(1)
        print("提取的json字符：{}".format(json_text))
        # 加载提取的字符串为JSON对象
        try:
            json_object = json.loads(json_text)
            print("generated results: {}".format(
                json.dumps(json_object,
                           indent=2,
                           ensure_ascii=False)))
            return json_object
        except json.JSONDecodeError as e:
            print("无法解析JSON字符串: ", e)
    else:
        print("没有找到匹配的JSON字符串")


def generate_similar_demonstration(
        demonstration_text,
        demonstration_requirements,
        model_name,
        demonstration_generation_instruction=None,
        model_config=None):
    """
    :param demonstration_text:
    :param demonstration_requirements:
    :param model_name:
    :param demonstration_generation_instruction:
    :param model_config:
    """
    if demonstration_generation_instruction is not None:
        pass
    else:
        demonstration_generation_instruction = Default_Instruction_4_Demonstration_Generation
    prompt = demonstration_generation_instruction.format(demonstration=demonstration_text,
                                                         other_requirements=demonstration_requirements)
    prompt = message_formatting(system_prompt=None, query=prompt)
    results = call_llm_with_message(prompt, model=model_name, model_config=model_config)
    return extract_from_markdown_json(results)


def example_expand(state, expand_config):
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
            and "demonstration_text" in expand_config.keys()
            and "demonstration_requirements" in expand_config.keys()), ("expand_config must contain \"num_expand\", "
                                                                        "\"model_name\", "
                                                                        "\"demonstration_generation_instruction\", "
                                                                        "\"demonstration_text\","
                                                                        "\"demonstration_requirements\"")

    expand = []
    num_expand = expand_config["num_expand"]
    model_name = expand_config["model_name"]
    demonstration_generation_instruction = expand_config["demonstration_generation_instruction"]
    demonstration_text = state[-1],
    demonstration_requirements = expand_config["demonstration_requirements"],
    # based on the current demonstration state[-1], expand num_expand demonstrations
    for _ in range(num_expand):
        state_copy = copy.deepcopy(state)
        new_demonstration = generate_similar_demonstration(
            demonstration_text=demonstration_text,
            demonstration_requirements=demonstration_requirements,
            model_name=model_name,
            demonstration_generation_instruction=demonstration_generation_instruction)
        print("new_demonstration: \n{}".format(new_demonstration))
        state_copy.append(new_demonstration)
        expand.append(state_copy)
    return expand


def demonstration_var_score(state):
    if len(state) == 1:
        return 0
    text_rerank_results = text_rerank(query=state[-1], documents=state[:-1])
    scores = [item["relevance_score"] for item in text_rerank_results["output"]["results"]]
    if len(scores) == 1:
        return [1 - scores[0]]
    else:
        return np.var(scores)


def beam_search(initial_state, max_steps, beam_width, expand_fn, score_fn, expand_fn_config):
    """
    Performs beam search.

    :param initial_state: The initial state to begin search from.
    :param max_steps: The maximum number of steps (or iterations) to perform.
    :param beam_width: The number of paths to explore at each step.
    :param expand_fn: A function that takes a state and returns a list of possible next states.
    :param score_fn: A function that takes a state and returns a score. Higher scores are better.
    :return: The best state found according to the scoring function.
    """

    # Initialize the beam with the initial state.
    beam = [(initial_state, score_fn(initial_state))]

    for step in range(max_steps):
        # Expand states in the current beam.
        candidates = []
        for state, score in beam:
            next_states = expand_fn(state, expand_fn_config)
            for next_state in next_states:
                next_score = score_fn(next_state)
                candidates.append((next_state, next_score))

        # Select the top `beam_width` states for the next iteration.
        print(candidates)
        candidates.sort(key=lambda x: x[1], reverse=True)
        beam = candidates[:beam_width]

    # Return the state with the highest score after the final iteration.
    best_state, best_score = max(beam, key=lambda x: x[1])
    return best_state


def demonstration_score_fn(state):
    # Score the state. In a real application, this score could be based on
    # model predictions, heuristics, or other criteria.
    # This toy example prefers states with more 'a's.
    return np.sum(state)


# def example_expand(state):
#     # Produce some new states from the current state.
#     # In a real application, this would involve generating possible next steps.
#     import copy
#     expand = []
#     for _ in range(3):
#         state_copy = copy.deepcopy(state)
#         state_copy.append(np.random.choice(5))
#         expand.append(state_copy)
#     return expand


if __name__ == '__main__':
    pass
    # model_name = "qwen2-57b-a14b-instruct"
    # demonstration_generation_instruction = Default_Instruction_4_Demonstration_Generation
    # demonstration_text = demonstration,
    # demonstration_requirements = other_requirements,
    # results = generate_similar_demonstration(demonstration_text=demonstration_text,
    #                                          demonstration_requirements=demonstration_requirements,
    #                                          model_name=model_name,
    #                                          demonstration_generation_instruction=demonstration_generation_instruction)
