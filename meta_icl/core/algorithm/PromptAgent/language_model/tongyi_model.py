import random
from http import HTTPStatus
from dashscope import Generation
import dashscope
import time

class TongyiModel():
    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float,
        **kwargs):
        
        if api_key is None:
            raise ValueError(f"api_key error: {api_key}")
        else:
            dashscope.api_key= api_key
        
        self.model_name = model_name
        self.temperature = temperature
        self.batch_forward_func = self.batch_forward_chatcompletion
        self.generate = self.gpt_chat_completion
    def batch_forward_chatcompletion(self, batch_prompts):
        """
        Input a batch of prompts to tongyi chat API and retrieve the answers.
        """
        responses = []
        for prompt in batch_prompts:
            response = self.gpt_chat_completion(prompt=prompt)
            responses.append(response)
        return responses
    
    def gpt_chat_completion(self, prompt):
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt}]
        backoff_time = 1
        while True:
            try:
                response = Generation.call(model=self.model_name,
                                messages=messages,
                                # 设置随机数种子seed，如果没有设置，则随机数种子默认为1234
                                seed=random.randint(1, 10000),
                                # 将输出设置为"message"格式
                                result_format='message',
                                temperature=self.temperature)
                if response.status_code == HTTPStatus.OK:
                    return response.output.choices[0].message.content.strip()
                else:
                    print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                        response.request_id, response.status_code,
                        response.code, response.message
                    ))
            except Exception as e:
                print(e, f' Sleeping {backoff_time} seconds...')
                time.sleep(backoff_time)
                backoff_time *= 1.5