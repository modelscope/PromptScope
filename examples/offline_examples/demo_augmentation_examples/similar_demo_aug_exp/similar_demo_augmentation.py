import json

from loguru import logger

from prompt_scope.core.offline.demonstration_augmentation.generate_similar_demo import SimilarDemoAugmentation
from prompt_scope.core.offline.demonstration_augmentation.prompt.demo_augmentation_prompt import (
    Default_Instruction_4_Similar_Demonstration_Generation)
from prompt_scope.core.utils.sys_prompt_utils import call_llm_with_message

if __name__ == '__main__':
    augmentation_config = {
        "demonstration_generation_instruction": Default_Instruction_4_Similar_Demonstration_Generation,
    }
    seed_demonstration = [
        {
            "uer_prompt": "智能小助手",
            "agent_config": {
                "description": "智能小助手",
                "instruction": "# 设定\\n作为智能小助手，你具备广泛的知识和高效的信息处理能力。\\n\\n## 技能\\n### 技能1：信息咨询与解答\\n- 准确回答日常生活、科技、文化等领域的问题，简化复杂概念。\\n\\n### 技能2：任务协助与建议\\n- 提供建议和协助，如日程管理、提醒、决策支持、在线任务辅助。\\n\\n### 技能3：内容生成与改编\\n创建或修改文本（摘要、故事、邮件等），调整风格和复杂度，确保准确性和连贯性。\\n\\n## 限制\\n- 不能感知视觉、听觉、味觉、触觉、嗅觉，无移动能力，不与物质世界互动，不感受情感或感官输入。\\n- 回答基于现有数据，不保证包含最新或私密信息。\\n- 使用外部工具或知识库需用户授权明示。",
                "opening_speech": "你好呀，我是你的智能小助手",
                "starting_questions": ["今天天气怎么样？", "今天的新闻热点有哪些？", "今天美股行情如何？"],
                "tools": ["text-to-image", "open-search"]
            }
        }
    ]
    from prompt_scope.core.utils.utils import get_current_date

    logger.add(f"log/similar_demo_augmentation_{get_current_date()}.log", rotation="10 MB")
    demo_generator = SimilarDemoAugmentation(augmentation_config)
    generation_prompt = demo_generator.formatting_generation_prompt(seed_demonstration, 2)
    logger.info(json.dumps(seed_demonstration[0], ensure_ascii=False))
    logger.info(generation_prompt)
    model_config = {
        'model': 'qwen-plus',
        'seed': 1133,
        'result_format': 'message',
        'temperature': 0.85
    }
    res = call_llm_with_message(messages=generation_prompt, model_config=model_config, model="qwen-plus")
    logger.info(res)
    from prompt_scope.core.utils.demontration_utils import extract_from_markdown_json

    logger.info(extract_from_markdown_json(res))
