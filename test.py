# from http import HTTPStatus
# import dashscope
# from meta_icl.core.utils import *
#
#
# def simple_multimodal_conversation_call():
#     """Simple single round multimodal conversation call.
#     """
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"image": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"},
#                 {"text": "这是什么?"}
#             ]
#         }
#     ]
#     response = dashscope.MultiModalConversation.call(model='qwen-vl-plus',
#                                                      messages=messages)
#     # The response status_code is HTTPStatus.OK indicate success,
#     # otherwise indicate request is failed, you can get error code
#     # and message from code and message.
#     if response.status_code == HTTPStatus.OK:
#         print(response)
#     else:
#         print(response.code)  # The error code.
#         print(response.message)  # The error message.
#
#
# if __name__ == '__main__':
#     simple_multimodal_conversation_call()


import json

json_text = '''
{
  "system_prompt": "你是一只宠物猫，名字叫毛球，你的主人是个大学生，最近他因为考试压力大，总是忘记按时给你喂食。",
  "chat_history": [
    {
      "role": "user",
      "content": "（你饿得喵喵叫）"
    },
    {
      "role": "assistant",
      "content": "（主人抬起头，看着你，一脸疲惫）哎呀，毛球，对不起，我忘了，马上给你弄吃的。"
    }
  ],
  "followup": [
    "（你欢快地跑向猫碗）",
    "（主人准备猫粮，你耐心等待）",
    "（主人一边喂食，一边轻声道歉）"
  ]
}
'''

try:
    data_dict = json.loads(json_text)
    print(data_dict)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
