from typing import List, Union, Optional, Dict, Literal
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
import json
import torch

class Message(BaseModel):
    role: str
    content: str

class VLLMHandler(BaseModel):
    model_path: str
    max_tokens: int = Field(default=512)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.95)
    presence_penalty: float = Field(default=0.0)
    frequency_penalty: float = Field(default=0.0)
    stop_tokens: Optional[List[str]] = Field(default=None)
    llm: Optional[LLM] = None
    tensor_parallel_size: Optional[int] = 1

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        if self.tensor_parallel_size > 1:
            self.llm = LLM(model=self.model_path, tensor_parallel_size=self.tensor_parallel_size)
        else:
            self.llm = LLM(model=self.model_path)

    def _create_sampling_params(self) -> SamplingParams:
        """Create VLLM sampling parameters."""
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            stop=self.stop_tokens
        )
    
    def generate(self, messages: List[List[Dict]] | List[str] | List[Dict] | str, message_type: Literal["string", "message"] = "message", chat_template = None) -> str:
        """Generate response using VLLM."""
        if not chat_template:
            if message_type == 'string':
                try:          
                # Generate response
                    outputs = self.llm.generate(
                        prompts=messages,
                        sampling_params=self._create_sampling_params()
                    )
                    # Return the generated text
                    return [output.outputs[0].text for output in outputs]

                except Exception as e:
                    raise RuntimeError(f"Error generating response: {e}")
            else:
                try:          
                    # Generate response
                    outputs = self.llm.chat(
                        messages=messages,
                        sampling_params=self._create_sampling_params()
                    )
                    # Return the generated text
                    return [output.outputs[0].text for output in outputs]

                except Exception as e:
                    raise RuntimeError(f"Error generating response: {e}")
        else:
            if message_type == 'string':
                try:          
                # Generate response
                    outputs = self.llm.generate(
                        prompts=messages,
                        sampling_params=self._create_sampling_params(),
                        chat_template=chat_template
                    )
                    # Return the generated text
                    return [output.outputs[0].text for output in outputs]

                except Exception as e:
                    raise RuntimeError(f"Error generating response: {e}")
            else:
                try:          
                    # Generate response
                    outputs = self.llm.chat(
                        messages=messages,
                        sampling_params=self._create_sampling_params(),
                        chat_template=chat_template
                    )
                    # Return the generated text
                    return [output.outputs[0].text for output in outputs]

                except Exception as e:
                    raise RuntimeError(f"Error generating response: {e}")

# Example usage:
if __name__ == "__main__":
    # Initialize the handler
    handler = VLLMHandler(
        model_path="/mnt/data/yunze.gy/model_pth/Qwen2-72B-Instruct",
        max_tokens=200,
        temperature=0.0,
        tensor_parallel_size=4,
        stop_tokens=["<|eot_id|>","<|start_header_id|>"]
    )

    # Example with message list
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    response = handler.generate(messages=messages)
    print(f"Message list response: {response}")