from typing import List, Type

from pydantic import BaseModel

from prompt_scope.core.llms.structured_output.pydantic_io import GptJsonIO, JsonStringError
from prompt_scope.core.schemas.message import ChatMessage, ChatResponse


def _structure_output(txt, prompt, err_msg, run_gpt_fn, pydantic_cls):
    gpt_json_io = GptJsonIO(pydantic_cls)
    analyze_res = run_gpt_fn(
        txt,
        sys_prompt=prompt + gpt_json_io.format_instructions
    )

    try:
        obj, err_msg = gpt_json_io.generate_output_auto_repair(analyze_res, run_gpt_fn), ""
    except JsonStringError as e:
        obj, err_msg = None, err_msg

    return ChatResponse(
        message=ChatMessage(
            role="assistant",
            additional_kwargs={
                "parsed": obj
            }
        ),
        error_message=err_msg
    )


class GeneralStructuredOutputMixin():

    def structured_output(self, messages: List[ChatMessage], schema: Type[BaseModel], **kwargs) -> BaseModel:
        query = messages.pop(-1)

        def run_gpt_fn(user_prompt, sys_prompt):
            messages.append(ChatMessage(role="system", content=sys_prompt))
            messages.append(ChatMessage(role="user", content=user_prompt))
            return self.chat(messages, **kwargs).message.content

        response = _structure_output(
            txt=query.content,
            prompt="",
            err_msg="Cannot parse output",
            run_gpt_fn=run_gpt_fn,
            pydantic_cls=schema
        )

        return response
