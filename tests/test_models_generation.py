import unittest

from meta_icl.core.models.generation_model import GenerationModel, AioGenerationModel, OpenAIGenerationModel, OpenAIAioGenerationModel
from meta_icl.core.utils.logger import Logger
import asyncio
import openai

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

        openai_config = {
            "module_name": "openai_generation",
            "model_name": "gpt-4o-mini",
            "max_tokens": 200,
            "top_k": 1,
            "seed": 1234,
        }
        self.openai_llm = OpenAIGenerationModel(**openai_config)

        openai_async_config = {
            "module_name": "openai_aio_generation",
            "model_name": "gpt-4o-mini",
            "max_tokens": 200,
            "top_k": 1,
            "seed": 1234,
        }
        self.openai_async_llm = OpenAIAioGenerationModel(**openai_config)
        
    def test_llm_prompt(self):
        prompt = "你是谁？"
        ans = self.llm.call(stream=False, prompt=prompt, result_format='message')

    def test_openai_llm_prompt(self):
        prompt = "你是谁？"
        ans = self.openai_llm.call(stream=False, prompt=prompt)

    def test_llm_messages(self):
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '你是谁？'}]
        ans = self.llm.call(stream=False, messages=messages, result_format='message')
    
    def test_openai_llm_messages(self):
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '你是谁？'}]
        ans = self.openai_llm.call(stream=False, messages=messages)

    def test_async_llm_prompt(self):
        prompts = ["你是谁？", "你会什么？", "天气怎么样？"]
        responses = asyncio.run(self.async_llm.async_call(prompts=prompts, result_format='message'))

    def test_openai_async_llm_prompt(self):
        prompts = ["你是谁？", "你会什么？", "天气怎么样？"]
        responses = asyncio.run(self.openai_async_llm.async_call(prompts=prompts))

    def test_async_llm_messages(self):
        messages = [[{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '你是谁？'}],
                [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '你会什么？'}],
                [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '天气怎么样？'}]]
        responses = asyncio.run(self.async_llm.async_call(list_of_messages=messages, result_format='message'))

    def test_openai_async_llm_messages(self):
        messages = [[{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '你是谁？'}],
                [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '你会什么？'}],
                [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '天气怎么样？'}]]
        responses = asyncio.run(self.async_llm.async_call(list_of_messages=messages))

if __name__ == '__main__':
    # unittest.main()
    openai.api_key = ''
    openai_config = {
            "module_name": "openai_generation",
            "model_name": "gpt-4o-mini",
            "max_tokens": 200,
            "seed": 1234,
        }
    openai_llm = OpenAIGenerationModel(**openai_config)
    prompt = "Who are you？"
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}]
    ans = openai_llm.call(stream=False, messages=messages)
    print(ans)