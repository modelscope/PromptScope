import unittest

from meta_icl.core.models.generation_model import GenerationModel, AioGenerationModel
from meta_icl.core.utils.logger import Logger
import asyncio

class TestLLILLM(unittest.TestCase):
    """Tests for LlamaIndexGenerationModel"""

    def setUp(self):
        config = {
            "module_name": "generation",
            "model_name": "qwen-plus",
            "max_tokens": 2000,
            "top_k": 1,
            "seed": 1234,
        }
        self.llm = GenerationModel(**config)

        async_config = {
            "module_name": "aio_generation",
            "model_name": "qwen-plus",
            "max_tokens": 2000,
            "top_k": 1,
            "seed": 1234,
        }
        self.async_llm = AioGenerationModel(**async_config)
        
        self.logger = Logger.get_logger()
    def test_llm_prompt(self):
        prompt = "你是谁？"
        ans = self.llm.call(stream=False, prompt=prompt, result_format='message')
        self.logger.info(ans.output.text)

    def test_llm_messages(self):
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '你是谁？'}]
        ans = self.llm.call(stream=False, messages=messages, result_format='message')
        self.logger.info(ans.output.text)
    
    def test_async_llm_prompt(self):
        prompts = ["你是谁？", "你会什么？", "天气怎么样？"]
        responses = asyncio.run(self.async_llm.async_call(prompts=prompts, result_format='message'))
        self.logger.info([response.output.text for response in responses])

    def test_async_llm_messages(self):
        messages = [[{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '你是谁？'}],
                [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '你会什么？'}],
                [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '天气怎么样？'}]]
        responses = asyncio.run(self.async_llm.async_call(list_of_messages=messages, result_format='message'))
        self.logger.info([response.output.text for response in responses])

if __name__ == '__main__':
    # unittest.main()

    config = {
            "module_name": "generation",
            "model_name": "qwen-plus",
            "max_tokens": 2000,
            "top_k": 1,
            "seed": 1234,
        }
    llm = GenerationModel(**config)
    prompt = "你是谁？"
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '你是谁？'}]
    ans = llm.call(stream=False, messages=messages, result_format='message')
    print(ans)