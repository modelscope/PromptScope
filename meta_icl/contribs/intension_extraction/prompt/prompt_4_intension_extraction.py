from meta_icl.core.utils import load_json_file

Instruction_for_query_rewrite = """你是一个优秀的客服专员，你能根据用户与客服的历史对话背景和当前用户query来精准识别用户当前的意图。
请直接清晰完整地输出用户当前意图。

以下是一些例子：
"""


def formatting_query_rewrite(examples, cur_query):
    prompt = (Instruction_for_query_rewrite
              + '\n'.join(
                f'###历史对话###：{item["chat_history"]}\n'
                f'###用户query###：{item["user_query"]}\n'
                f'###用户意图###：{item["user_intention"]}\n\n'
                for item in examples))
    prompt += (
        f'请直接清晰完整地输出用户当前意图\n###历史对话###：{cur_query["chat_history"]}\n###用户query###：{cur_query["user_query"]}\n'
        f'###用户意图###：')
    return prompt


# Instruction_for_intention_analysis = f'你是一个优秀的客服专员，你能根据用户与客服的历史对话背景和当前用户query来精准识别用户当前的意图。\n\n你的任务是：\n1. 请直接清晰完整地输出用户当前的意图。\n2. 请根据当前用户的意图，分类用户意图。可选的类别有:\n["课程答疑", "课程观看", "延期", "有效期咨询", "课程转移", "课程下载", "课程资料", "直播课领取", "购买咨询", "课程购买后操作", "学习效果反馈", "商品版权疑问", "发货咨询", "物流方式", "物流包邮", "物流运费险", "物流查询", "退款咨询", "要求退款", "引导", "商品开票", "课程优惠", "课程适用范围", "课程试听", "APP下载", "课程下载", "学浪","登录问题"]\n'
# Instruction_for_intention_analysis = f'你是一个优秀的客服专员，你能根据用户与客服的历史对话背景和当前用户query来精准识别用户当前的意图。\n\n你的任务是：\n1. 请直接清晰完整地输出用户当前的意图。\n2. 请根据当前用户的意图，分类用户意图。你的分类应该严格属于以下类别之一:\n["课程答疑", "课程观看", "延期", "有效期咨询", "课程转移", "课程下载", "课程资料", "直播课领取", "购买咨询", "课程购买后操作", "学习效果反馈", "商品版权疑问", "发货咨询", "物流方式", "物流包邮", "物流运费险", "物流查询", "退款咨询", "要求退款", "引导", "商品开票", "课程优惠", "课程适用范围", "课程试听", "APP下载", "课程下载", "学浪","登录问题"]\n'

INTENTION_CLASS_PTH="data/intention_classes.json"
INTENTION_CLASS = load_json_file(INTENTION_CLASS_PTH)
# Instruction_for_intention_analysis = f'你是一个优秀的客服专员，你能根据用户与客服的历史对话背景和当前用户query来精准识别用户当前的意图。\n\n你的任务是：\n1. 请直接清晰完整地输出用户当前的意图。请注意保留历史对话中的订单，手机号等关键信息。\n2. 请根据当前用户的意图，分类用户意图。你的分类应该严格属于以下类别之一:\n{INTENTION_CLASS}\n'

Instruction_for_intention_analysis = f'你是一个优秀的客服专员，你能根据用户与客服的历史对话背景和当前用户query来精准识别用户当前的意图。\n\n你的任务是：\n1. 请直接清晰完整地输出用户当前的意图。\n2. 请根据当前用户的意图，分类用户意图。你的分类应该严格属于以下类别之一:\n{INTENTION_CLASS}\n\n请注意：1. 在输出用户意图是，请保留历史对话中的关键信息，例如\na. 包含数字的手机号，订单号，\nb. 用户使用的平台,如小程序，抖音，获课app等，\nc. 历史对话中和当前用户输入相关的用户意图。\n\n'

def formatting_intention_classification(examples, cur_query, configs=None):
    prompt = (Instruction_for_intention_analysis + '\n\n以下是一些例子：\n'
              + '\n'.join(
                f'###历史对话###：{item["chat_history"]}\n'
                f'###用户query###：{item["user_query"]}\n'
                f'###用户意图分析###：{{"user_intention": \"{item["user_intention"]}\", "intention_class": \"{item["intention_class"]}\" }}'
                for item in examples))
    prompt += (
        f'\n请根据提供的历史对话和用户query，直接清晰完整地输出用户当前意图\n###历史对话###：{cur_query["chat_history"]}\n###用户query###：{cur_query["user_query"]}\n'
        f'###用户意图分析###：')
    return prompt
