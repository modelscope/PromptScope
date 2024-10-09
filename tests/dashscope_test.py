import unittest
from unittest.mock import MagicMock, patch
import unittest.mock as mock
import dashscope

from meta_icl.core.models.generation_model import GenerationModel, AioGenerationModel
from meta_icl.core.scheme.model_response import ModelResponse, ModelResponseGen

class TestGenerationModel(unittest.TestCase):
    """Tests for GenerationModel"""

    def setUp(self):
        config = {
            "module_name": "generation",
            "model_name": "qwen-plus",
            "max_tokens": 2000,
            "top_k": 1,
            "seed": 1234,
        }
        self.generation_model = GenerationModel(**config)

    @patch("meta_icl.core.models.generation_model.GenerationModel._call")
    def test_llm_prompt(self, mock_generation_call: MagicMock):
        # Set up the mock response for a successful API call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.request_id = "test_request_id"
        mock_response.usage = {
            "input_tokens": 3,
            "output_tokens": 5,
            "total_tokens": 8,
        }
        # mock_response.output = dashscope.api_entities.dashscope_response.GenerationOutput(
        #     text = "Hello!"
        # )
        mock_response.output.text = "Hello!"
        mock_response.output.choices = []
        mock_generation_call.return_value = mock_response

        prompt = "Hi!"
        response = self.generation_model.call(stream=False, prompt=prompt)
        # Verify the response
        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(response.message.content, "Hello!")
        self.assertEqual(response.raw, mock_response)

        # Verify call
        mock_generation_call.assert_called_once_with(
            stream=False,
            messages=[],
            prompt=prompt,
            **{"max_tokens": 2000,
            "top_k": 1,
            "seed": 1234},
        )

        with self.assertRaises(ValueError) as cm:
            self.generation_model.call(prompt = "Who are you？", messages =[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi!"},])
        # Prompt and messages can not be passed together
        self.assertEqual(
            "prompt and messages cannot be both specified",
            str(cm.exception)
        )

    @patch("meta_icl.core.models.generation_model.GenerationModel._call")
    def test_llm_messages(self, mock_generation_call: MagicMock):
        # Set up the mock response for a successful API call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.request_id = "test_request_id"
        mock_response.usage = {
            "input_tokens": 3,
            "output_tokens": 5,
            "total_tokens": 8,
        }
        mock_response.output.choices = [dashscope.api_entities.dashscope_response.Choice(
            finish_reason='stop',
            message={'role': 'assistant', 'content': 'Hello!'})]
        mock_response.output.text = ''
        mock_generation_call.return_value = mock_response

        # Define test input
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi!"},
        ]
        response = self.generation_model.call(stream=False, messages=messages, result_format='message')
        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(response.message.content, "Hello!")
        self.assertEqual(response.raw, mock_response)

        mock_generation_call.assert_called_once_with(
            stream=False,
            messages=messages,
            prompt='',
            **{"max_tokens": 2000,
            "result_format": "message",
            "top_k": 1,
            "seed": 1234},
        )
        # Prompt and messages can not be passed together
        with self.assertRaises(ValueError) as cm:
            self.generation_model.call(prompt = "Who are you？", messages =[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi!"},])
        # Prompt and messages can not be passed together
        self.assertEqual(
            "prompt and messages cannot be both specified",
            str(cm.exception)
        )

class TestAioGenerationModel(unittest.IsolatedAsyncioTestCase):
    """Tests for AioGenerationModel"""

    def setUp(self):
        async_config = {
                "module_name": "aio_generation",
                "model_name": "qwen-plus",
                "max_tokens": 2000,
                "top_k": 1,
                "seed": 1234,
            }
        self.aio_generation_model = AioGenerationModel(**async_config)

    @patch("meta_icl.core.models.generation_model.AioGenerationModel._async_call")
    async def test_async_llm_prompt(self, mock_generation_call: MagicMock):
        # Set up the mock response for a successful API call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.request_id = "test_request_id"
        mock_response.usage = {
            "input_tokens": 3,
            "output_tokens": 5,
            "total_tokens": 8,
        }
        mock_response.output.text = "Hello!"
        mock_response.output.choices = []
        mock_generation_call.return_value = mock_response

        prompts = ["Hello!", "Hi!", "How are you?"]
        responses = await self.aio_generation_model.async_call(prompts=prompts)
        for response in responses:
            self.assertIsInstance(response, ModelResponse)
            self.assertEqual(response.message.content, "Hello!")
            self.assertEqual(response.raw, mock_response)

        mock_generation_call.assert_has_calls(
            [
            mock.call(
                prompt=prompt,
                messages=[],
                **{"max_tokens": 2000,
                    "top_k": 1,
                    "seed": 1234},
                    )
            for prompt in prompts
            ], any_order=True
        )

        with self.assertRaises(ValueError) as cm:
            await self.aio_generation_model.async_call(
                prompts=["Hello!", "Hi!", "How are you?"],
                list_of_messages=[[{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Hello!'}],
                [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Hi!'}],
                [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'How are you？'}]])
        # Prompt and messages can not be passed together
        self.assertEqual(
            "prompt and messages cannot be both specified",
            str(cm.exception)
        )

    @patch("meta_icl.core.models.generation_model.AioGenerationModel._async_call")
    async def test_async_llm_messages(self, mock_generation_call: MagicMock):
        # Set up the mock response for a successful API call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.request_id = "test_request_id"
        mock_response.usage = {
            "input_tokens": 3,
            "output_tokens": 5,
            "total_tokens": 8,
        }
        mock_response.output.text = ''
        mock_response.output.choices = [dashscope.api_entities.dashscope_response.Choice(
            finish_reason='stop',
            message={'role': 'assistant', 'content': 'Hello!'})]
        mock_generation_call.return_value = mock_response
        mock_generation_call.return_value = mock_response
        list_of_messages = [[{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Hello!'}],
                [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Hi!'}],
                [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'How are you？'}]]
        responses = await self.aio_generation_model.async_call(list_of_messages=list_of_messages)
        for response in responses:
            self.assertIsInstance(response, ModelResponse)
            self.assertEqual(response.message.content, "Hello!")
            self.assertEqual(response.raw, mock_response)

        mock_generation_call.assert_has_calls(
            [
            mock.call(
                prompt='',
                messages=messages,
                **{"max_tokens": 2000,
                    "top_k": 1,
                    "seed": 1234},
                    )
            for messages in list_of_messages
            ], any_order=True
        )

        with self.assertRaises(ValueError) as cm:
            await self.aio_generation_model.async_call(
                prompts=["Hello!", "Hi!", "How are you?"],
                list_of_messages=[[{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Hello!'}],
                [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Hi!'}],
                [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'How are you？'}]])
        # Prompt and messages can not be passed together
        self.assertEqual(
            "prompt and messages cannot be both specified",
            str(cm.exception)
        )

if __name__ == '__main__':
    unittest.main()
