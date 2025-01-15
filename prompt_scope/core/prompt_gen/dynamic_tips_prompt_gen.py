from typing import Dict

from prompt_scope.core.prompt_gen.prompt_gen import BasePromptGen
from prompt_scope.core.retriever.tips_retriever import TipsRetriever




class DynamicTipsPromptGen(BasePromptGen):
    tips_retriever: TipsRetriever

    def generate(self, llm_input: Dict[str, str], search_tips_query: str = None, top_k: int = 1, threshold: float = 0.5):
        # retrieve tips
        tips_list = self.tips_retriever.invoke(search_tips_query, top_k, threshold)
        # add tips into prompt
        llm_input["<<tips>>"] = self.process_tips(tips_list)
        # invoke llm
        messages = super()._messages_translator(llm_input)
        llm = super()._llm_translator()
        response = llm.chat(messages=messages).message.content
        return response

    def process_tips(self, tips_list):
        return "\n".join(tips_list)

    @staticmethod
    def load(promptgen_load_dir: str, tips_load_path: str = None):
        prompt_gen = BasePromptGen.load(promptgen_load_dir)

        tips_retriever = TipsRetriever.load(load_path=tips_load_path)

        return DynamicTipsPromptGen(
            llm_config=prompt_gen.llm_config,
            prompt_params=prompt_gen.prompt_params,
            tips_retriever=tips_retriever
        )


