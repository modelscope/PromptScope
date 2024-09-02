import unittest
from unittest.mock import MagicMock, patch
import openai
import unittest.mock as mock

from meta_icl.core.models.generation_model import OpenAIGenerationModel, OpenAIAioGenerationModel
from meta_icl.core.scheme.model_response import ModelResponse

class TestOpenAIGenerationModel(unittest.TestCase):
    """Tests for OpenAIGenerationModel"""

    def setUp(self):
        openai_config = {
            "module_name": "openai_generation",
            "model_name": "gpt-4o-mini",
            "max_tokens": 200,
            "top_k": 1,
            "seed": 1234,
        }
        self.openai_llm = OpenAIGenerationModel(**openai_config)

    @patch("meta_icl.core.models.generation_model.OpenAIGenerationModel._call")
    def test_openai_llm_prompt(self, mock_generation_call: MagicMock):
        mock_response = MagicMock()
        mock_response.id = "test_request_id"
        mock_response.usage = {
            "completion_tokens": 3,
            "prompt_tokens": 5,
            "total_tokens": 8,
        }
        mock_response.choices = [openai.types.chat.chat_completion.Choice(
            index=0,
            message={
                "role": "assistant",
                "content": "Hello!"
                },
                finish_reason="stop"
                )]
        mock_response.status_code = 200
        mock_generation_call.return_value = mock_response

        prompt = "Hi!"
        response = self.openai_llm.call(stream=False, prompt=prompt)

        # Verify the response
        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(response.message.content, "Hello!")
        self.assertEqual(response.raw, mock_response)

        # Verify call
        mock_generation_call.assert_called_once_with(
            stream=False,
            prompt=prompt,
            messages=[],
            **{"max_tokens": 200,
            "top_k": 1,
            "seed": 1234},
        )

        with self.assertRaises(ValueError) as cm:
            self.openai_llm.call(prompt = "Who are you？", messages =[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi!"},])
        # Prompt and messages can not be passed together
        self.assertEqual(
            "prompt and messages cannot be both specified",
            str(cm.exception)
        )
    
    @patch("meta_icl.core.models.generation_model.OpenAIGenerationModel._call")
    def test_openai_llm_messages(self, mock_generation_call: MagicMock):
        mock_response = MagicMock()
        mock_response.id = "test_request_id"
        mock_response.usage = {
            "completion_tokens": 3,
            "prompt_tokens": 5,
            "total_tokens": 8,
        }
        mock_response.choices = [openai.types.chat.chat_completion.Choice(
            index=0,
            message={
                "role": "assistant",
                "content": "Hello!"
                },
                finish_reason="stop"
                )]
        mock_response.status_code = 200
        mock_generation_call.return_value = mock_response

        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Hi!'}]
        response = self.openai_llm.call(stream=False, messages=messages)

        # Verify the response
        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(response.message.content, "Hello!")
        self.assertEqual(response.raw, mock_response)

        # Verify call
        mock_generation_call.assert_called_once_with(
            stream=False,
            prompt='',
            messages=messages,
            **{"max_tokens": 200,
            "top_k": 1,
            "seed": 1234},
        )

        with self.assertRaises(ValueError) as cm:
            self.openai_llm.call(prompt = "Who are you？", messages =[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi!"},])
        # Prompt and messages can not be passed together
        self.assertEqual(
            "prompt and messages cannot be both specified",
            str(cm.exception)
        )

class TestOpenAIAioGenerationModel(unittest.IsolatedAsyncioTestCase):
    """Tests for OpenAIAioGenerationModel"""

    def setUp(self):
        openai_async_config = {
            "module_name": "openai_aio_generation",
            "model_name": "gpt-4o-mini",
            "max_tokens": 200,
            "top_k": 1,
            "seed": 1234,
        }
        self.openai_async_llm = OpenAIAioGenerationModel(**openai_async_config)
    @patch("meta_icl.core.models.generation_model.OpenAIAioGenerationModel._async_call")
    async def test_openai_async_llm_prompt(self, mock_generation_call):
        mock_response = MagicMock()
        mock_response.id = "test_request_id"
        mock_response.usage = {
            "completion_tokens": 3,
            "prompt_tokens": 5,
            "total_tokens": 8,
        }
        mock_response.choices = [openai.types.chat.chat_completion.Choice(
            index=0,
            message={
                "role": "assistant",
                "content": "Hello!"
                },
                finish_reason="stop"
                )]
        mock_response.status_code = 200
        mock_generation_call.return_value = mock_response

        prompts = ["Hello!", "Hi!", "How are you?"]
        responses = await self.openai_async_llm.async_call(stream=False, prompts=prompts)
        for response in responses:
            self.assertIsInstance(response, ModelResponse)
            self.assertEqual(response.message.content, "Hello!")
            self.assertEqual(response.raw, mock_response)

        # Verify call
        mock_generation_call.assert_has_calls(
            [
            mock.call(
                stream=False,
                prompt=prompt,
                messages=[],
                **{"max_tokens": 200,
                    "top_k": 1,
                    "seed": 1234},
                    ) 
            for prompt in prompts
            ], any_order=True
        )

        with self.assertRaises(ValueError) as cm:
            await self.openai_async_llm.async_call(
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

    @patch("meta_icl.core.models.generation_model.OpenAIAioGenerationModel._async_call")
    async def test_openai_async_llm_messages(self, mock_generation_call):
        mock_response = MagicMock()
        mock_response.id = "test_request_id"
        mock_response.usage = {
            "completion_tokens": 3,
            "prompt_tokens": 5,
            "total_tokens": 8,
        }
        mock_response.choices = [openai.types.chat.chat_completion.Choice(
            index=0,
            message={
                "role": "assistant",
                "content": "Hello!"
                },
                finish_reason="stop"
                )]
        mock_response.status_code = 200
        mock_generation_call.return_value = mock_response

        list_of_messages = [[{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Hello!'}],
                [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Hi!'}],
                [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'How are you？'}]]
        responses = await self.openai_async_llm.async_call(list_of_messages=list_of_messages)
        for response in responses:
            self.assertIsInstance(response, ModelResponse)
            self.assertEqual(response.message.content, "Hello!")
            self.assertEqual(response.raw, mock_response)

        mock_generation_call.assert_has_calls(
            [
            mock.call(
                prompt='',
                messages=messages,
                **{"max_tokens": 200,
                    "top_k": 1,
                    "seed": 1234},
                    ) 
            for messages in list_of_messages
            ], any_order=True
        )

        with self.assertRaises(ValueError) as cm:
            await self.openai_async_llm.async_call(
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