import unittest

from meta_icl.core.models.generation_model import LlamaIndexGenerationModel, AioGenerationModel
from meta_icl.core.utils.logger import Logger
import time
import random
from http import HTTPStatus
from dashscope import Generation  # 建议dashscope SDK 的版本 >= 1.14.0
import dashscope
import yaml
import os
import asyncio
import aiohttp
from meta_icl.core.scheme.model_response import ModelResponse
from meta_icl.core.utils.timer import timer

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
    
    async def test_llm_async_call(self):
        prompts = ["斗破苍穹的作者是？", "斗罗大陆的作者是？"]
        # start_time = time.time()
        anses = []
        for prompt in prompts:
            ans = await self.llm.async_call(prompt=prompt)
            anses.append(ans)
        # end_time = time.time()
        # print("time:", end_time - start_time)
        self.logger.info(anses.message.content)

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

def test_async():
    prompts = ["你是谁？", "你会什么？", "天气怎么样？"]
    prompt = "斗破苍穹的作者是？"

    config = {
            "module_name": "aio_generation",
            "model_name": "qwen2-57b-a14b-instruct",
            "max_tokens": 2000,
            "seed": 1234,
        }
    llm = AioGenerationModel(**config)
    # total_t = 0
    responses = asyncio.run(llm.async_call(prompt=prompts))
    # responses = []
    # for prompt in prompts:
    #     response, t = llm.call(prompt=prompt)
    #     responses.append(response)
    #     print(response, t)
    # print(t)
    print([response.output.text for response in responses])

if __name__ == '__main__':
    # unittest.main()
    test_async()
    # response = llm.call(prompt=prompt)
    # print(response)
    # exit()
    # async def test_llm_async_call(llm, prompts, retry=5):
    #     semaphore = asyncio.Semaphore(2)
    #     print(f'running!')
    #     async def single_run(prompt, index):
    #         if prompt == None:
    #             return
    #         async with semaphore:
    #             ret_json = await llm.async_call(prompt=prompt)
    #             return {'content': ret_json,
    #                     'index': index}
    #     responses = [
    #         asyncio.create_task(single_run(prompts[i], i))
    #         for i in range(len(prompts))
    #     ]
    #     return await asyncio.gather(*responses)
    
    
    # end_time = time.time()
    # print("time:", end_time - start_time)
    # prompt = "斗破苍穹的作者是？仅输出作者名字，不需要其他信息。"
    # start_time = time.time()
    # ans = llm.call(stream=False, prompt=prompt).message.content
    
    # # ans = call_llm(prompt, "qwen2-57b-a14b-instruct", temperature=0.85)

    # print(ans)
    # end_time = time.time()
    # print("time:", end_time - start_time)
