import unittest

from meta_icl.core.models.generation_model import LlamaIndexGenerationModel
from meta_icl.core.utils.logger import Logger

class TestLLILLM(unittest.TestCase):
    """Tests for LlamaIndexGenerationModel"""

    def setUp(self):
        config = {
            "module_name": "dashscope_generation",
            "model_name": "qwen-plus",
            "clazz": "models.llama_index_generation_model",
            "max_tokens": 2000,
            "top_k": 1,
            "seed": 1234,
        }
        self.llm = LlamaIndexGenerationModel(**config)
        self.logger = Logger.get_logger()
    def test_llm_prompt(self):
        prompt = "斗破苍穹的作者是？"
        ans = self.llm.call(stream=False, prompt=prompt)
        self.logger.info(ans.message.content)

if __name__ == '__main__':
    unittest.main()
