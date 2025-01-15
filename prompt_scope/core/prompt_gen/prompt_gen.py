import os
import yaml
import json
from typing import Dict
from pydantic import BaseModel
from abc import abstractmethod, ABC

from ..utils.utils import load_yaml
from ..utils.text_utils import is_chinese_prompt
from ..llms.dashscope_llm import DashscopeLLM
from ..prompt_gen.base import LLMConfig, PromptParams
from ..schemas.message import ChatMessage, MessageRole
from prompt_scope.core.llms.base import BaseLLM


class BasePromptGen(BaseModel, ABC):
    prompt_params: PromptParams = None

    _is_chinese_prompt: bool = False

    def _messages_translator(self, llm_input):
        system_prompt = self.prompt_params.system_prompt
        user_prompt = self.prompt_params.user_prompt
        examples = self.prompt_params.examples

        system_user_prompt = system_prompt+user_prompt if system_prompt is not None else user_prompt
        self._is_chinese_prompt = is_chinese_prompt(system_user_prompt)

        for ph in llm_input.keys():
            flag = False
            if system_prompt is not None and ph in system_prompt:
                system_prompt = system_prompt.replace(ph, llm_input[ph])
                flag = True

            if ph in user_prompt:
                user_prompt = user_prompt.replace(ph, llm_input[ph])
                flag = True

            if not flag:
                raise RuntimeError(f"{ph} not found in system or user")

        if examples is not None:
            examples_str = self._transform_exemplar_to_str(examples)
            if system_prompt is not None and len(system_prompt.strip())>0:
                system_prompt += f"\n\n{examples_str}"
            else:
                user_prompt += f"\n\n{examples_str}"

        messages = []
        if system_prompt is not None and len(system_prompt.strip()) > 0:
            messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))
        messages.append(ChatMessage(role=MessageRole.USER, content=user_prompt))

        return messages

    def _transform_exemplar_to_str(self, examples):
        example_str_list = []
        for example in examples:
            llm_input_str = ""
            for key, value in example.llm_input.items():
                llm_input_str += f"{key}: {value}\n"

            llm_output_str = example.llm_output

            if self._is_chinese_prompt:
                example_str_list.append(f"输入：\n{llm_input_str}\n输出：\n{llm_output_str}")
            else:
                example_str_list.append(f"Input:\n{llm_input_str}\nOutput:\n{llm_output_str}")

        examples_str = "\n\n".join(example_str_list)

        if self._is_chinese_prompt:
            return f"以下是一些参考示例：\n{examples_str}"
        else:
            return f"Here are some reference examples:\n{examples_str}"

    def _llm_translator(self):
        client = self.llm_config.client
        model = self.llm_config.model
        llm = DashscopeLLM(client=client, model=model)
        return llm

    def generate(self, llm, llm_input: Dict[str, str], **kwargs):
        messages = self._messages_translator(llm_input)
        response = llm.chat(messages=messages).message.content
        return response

    @staticmethod
    def load(promptgen_load_dir):

        def _load_prompt_params(prompt_params_load_path):
            with open(prompt_params_load_path, "r") as f:
                system_prompt = None
                user_prompt = None
                placeholders = []
                examples = None

                for line in f:
                    if line.startswith("<system>"):
                        system_prompt = ""
                        for line in f:
                            if line.startswith("</system>"):
                                break
                            system_prompt += line

                    if line.startswith("<user>"):
                        user_prompt = ""
                        for line in f:
                            if line.startswith("</user>"):
                                break
                            user_prompt += line

                    if line.startswith("<examples>"):
                        examples = ""
                        for line in f:
                            if line.startswith("</examples>"):
                                break
                            examples += line

            system_prompt = system_prompt.strip() if system_prompt is not None else None
            user_prompt = user_prompt.strip()
            if examples is not None:
                examples = json.loads(examples)

            return PromptParams(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                placeholders=placeholders,
                examples=examples
            )

        prompt_params = _load_prompt_params(prompt_params_load_path=f"{promptgen_load_dir}/prompt.param")

        return BasePromptGen(prompt_params=prompt_params)

    def save(self,  save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        llm_config_save_path = f"{save_dir}/llm_config.yml"

        llm_config = self.llm_config.model_dump()
        save_llm_config = {}
        for key, value in llm_config.items():
            if key == "api_key" or value is None:
                continue
            save_llm_config[key] = value

        with open(llm_config_save_path, "w") as f:
            yaml.dump(save_llm_config, f)

        prompt_params_save_path = f"{save_dir}/prompt.param"
        plain_prompt_text = ""
        if self.prompt_params.system_prompt is not None:
            plain_prompt_text += f"<system>\n{self.prompt_params.system_prompt}\n</system>\n\n"

        plain_prompt_text += f"<user>\n{self.prompt_params.user_prompt}\n</user>\n\n"
        placeholders = "\n".join(self.prompt_params.placeholders)
        plain_prompt_text += f"<placeholders>\n{placeholders}\n</placeholders>\n\n"

        examples = [e.model_dump() for e in self.prompt_params.examples]
        examples_str = json.dumps(examples, ensure_ascii=False, indent=4)
        plain_prompt_text += f"<examples>\n{examples_str}\n</examples>"

        with open(prompt_params_save_path, "w") as f:
            f.write(plain_prompt_text)


    @property
    def raw_text(self):
        raw_text = ""
        if self.prompt_params.system_prompt is not None:
            raw_text += f"[SYSTEM]\n{self.prompt_params.system_prompt}\n\n"

        raw_text += f"[USER]\n{self.prompt_params.user_prompt}"

        if self.prompt_params.examples is not None:
            examples_str = self._transform_exemplar_to_str(self.prompt_params.examples)
            raw_text += f"\n\n[EXAMPLES]\n{examples_str}\n"

        return raw_text

