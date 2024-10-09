import os
import random
from http import HTTPStatus

import dashscope
import yaml
from dashscope import Generation  # 建议dashscope SDK 的版本 >= 1.14.0

KEY = ""
dashscope.api_key = KEY


def call_llm(prompt, model_name, temperature=1.0):
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}]
    response = Generation.call(model=model_name,
                               messages=messages,
                               # 设置随机数种子seed，如果没有设置，则随机数种子默认为1234
                               seed=random.randint(1, 10000),
                               # 将输出设置为"message"格式
                               result_format='message',
                               temperature=temperature)
    if response.status_code == HTTPStatus.OK:
        # print(response)
        try:
            return response.output.choices[0].message.content
        except:
            return response.output.choices[0].output.text
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))


def prompt_rewrite(query):
    templates = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'templates.yaml'), 'r'))
    template_names = ['CLEVER', 'ICIO', 'CRISPE', 'BROKE', 'RASCEF']
    result = {}
    for name in template_names:
        template = templates[name]
        prompt = template.format(query=query)
        # prompt_rewrite(prompt)
        result[name] = call_llm(prompt, "qwen2-57b-a14b-instruct")
    return result


def prompt_evaluation(prompts, ranking_prompt, query):
    prompt_templates = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'templates.yaml'), 'r'))
    eval_prompt = prompt_templates['Evaluation']
    contents = []
    for name, prompt in prompts.items():
        prompt = "|".join([prompt, query])
        print("##############", prompt, "###################")
        content = call_llm(prompt, "qwen2-57b-a14b-instruct")
        contents.append(content)
    answers = "｜".join(contents)
    print(answers)
    eval_prompt = eval_prompt.format(ranking_prompt=ranking_prompt, answers=answers)
    print(eval_prompt)
    # prompt_rewrite(prompt)
    response = call_llm(eval_prompt, "qwen2-57b-a14b-instruct", temperature=0.1)
    print(response)
    return response


def generate_query(prompt):
    prompt_templates = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'templates.yaml'), 'r'))
    query_generation = prompt_templates['Query_Generation'].format(prompt=prompt)
    query = call_llm(query_generation, "qwen2-57b-a14b-instruct")
    return query
