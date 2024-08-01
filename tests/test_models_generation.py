import unittest

from meta_icl.core.models.generation_model import LlamaIndexGenerationModel
from meta_icl.core.utils.logger import Logger
import time
import random
from http import HTTPStatus
from dashscope import Generation  # 建议dashscope SDK 的版本 >= 1.14.0
import dashscope
import yaml
import os

class TestLLILLM(unittest.TestCase):
    """Tests for LlamaIndexGenerationModel"""

    def setUp(self):
        config = {
            "module_name": "dashscope_generation",
            "model_name": "qwen-plus",
            "clazz": "models.llama_index_generation_model",
            "max_tokens": 2000,
            "top_k": 1,
            "seed": 1234,
        }
        self.llm = LlamaIndexGenerationModel(**config)
        self.logger = Logger.get_logger()
    def test_llm_prompt(self):
        prompt = "斗破苍穹的作者是？"
        # start_time = time.time()
        ans = self.llm.call(stream=False, prompt=prompt)
        # end_time = time.time()
        # print("time:", end_time - start_time)
        self.logger.info(ans.message.content)

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
        return response.output.choices[0].message.content
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
if __name__ == '__main__':
    unittest.main()
    # config = {
    #         "module_name": "dashscope_generation",
    #         "model_name": "qwen2-57b-a14b-instruct",
    #         "clazz": "models.llama_index_generation_model",
    #         "max_tokens": 2000,
    #         "top_k": 1,
    #         "seed": 1234,
    #     }
    # llm = LlamaIndexGenerationModel(**config)
    # prompt = "斗破苍穹的作者是？仅输出作者名字，不需要其他信息。"
    # start_time = time.time()
    # ans = llm.call(stream=False, prompt=prompt).message.content
    
    # # ans = call_llm(prompt, "qwen2-57b-a14b-instruct", temperature=0.85)

    # print(ans)
    # end_time = time.time()
    # print("time:", end_time - start_time)
