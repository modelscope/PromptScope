from abc import ABC, abstractmethod

from meta_icl.core.utils.sys_prompt_utils import check_dir, sav_json
from meta_icl.core.utils.utils import get_current_date
import json, os
from typing import List, Any
import numpy as np

from loguru import logger
from meta_icl.core.models.generation_model import GenerationModel
from meta_icl.core.utils.utils import extract_from_markdown_json
from meta_icl.core.utils.sys_prompt_utils import (call_llm_with_message, message_formatting, text_rerank,
                                                  convert_model_name_to_model_config)
from meta_icl.core.utils.demontration_utils import demo_augmentation_by_llm_prompt_org
from loguru import logger
from typing import Union, Dict, List
from meta_icl.core.utils.prompt_handler import PromptHandler
from meta_icl.core.enumeration.language_enum import LanguageEnum

# Default_Instruction_4_Demonstration_Generation = """请根据提供的样例，给出${num_generated_examples}个类似样例，要求和现在的样例的任务类型一致。
#
# 要求：
# 1. 生成的语言和提供的参考样例保持一致， 即提供的参考样例是英文的，你给出的样例也应该是英文的；如果提供的参考样例是中文的，你给出的样例也应该是中文的
# 2. 给出的样例尽量与参考样例属于同一个任务类型，但和参考样例有较大区别，并且是不同domain的。
# 3. 和提供的参考样例保持一致输出格式，并且每个样例用markdown json 形式单独区分。
# ${other_requirements}
#
# 参考样例：
# ```json
# ${demonstration}
# ```
#
# 请给出${num_generated_examples}个类似样例:
# """
from meta_icl.core.offline.demonstration_augmentation.base_demo_augmention import BaseDemonstrationAugmentation


class BaseAugmentationByBeamSearch(BaseDemonstrationAugmentation):
    @abstractmethod
    def expand_fn(self, **kwargs):
        pass

    @abstractmethod
    def score_fn(self, **kwargs):
        pass

    @abstractmethod
    def beam_search_generation(self, **kwargs):
        pass

    # @abstractmethod
    # def run(self, seed_demonstrations: Union[str, List[str], Dict, Any],
    #         n: int, **kwargs) -> List:
    #     pass


class BeamSearchGenerationByDiversity(BaseAugmentationByBeamSearch):
    FILE_PATH: str = __file__

    def __init__(self,
                 demonstration_save_dir,
                 num_expand: int,
                 demonstration_generation_instruction: str = "",
                 language: str = "cn",
                 prompt_handler: PromptHandler = None,
                 demonstration_requirements: str = "",
                 auto_save=True,
                 expand_model_config=None,
                 model_name=None,
                 class_name: str = "",
                 prompt_file: str = "",
                 prompt_dict: dict = None,
                 ):
        """
        # :param demonstration_var_score
        # :param demonstration_expand: func
        :param demonstration_save_dir: func
        """

        super().__init__()
        self._prompt_handler = prompt_handler
        self.model = None
        check_dir(demonstration_save_dir)
        self.demonstration_save_dir = demonstration_save_dir
        self.all_states = []
        self.auto_save = auto_save

        self.num_expand = num_expand
        self._demonstration_generation_instruction = demonstration_generation_instruction
        self.demonstration_requirements = demonstration_requirements
        self.expand_model_config = expand_model_config
        self.model_name = model_name

        self.language = LanguageEnum(language)
        self.class_name = class_name,
        self.prompt_file = prompt_file,
        self.prompt_dict = prompt_dict
    @property
    def demonstration_generation_instruction(self):
        if self._demonstration_generation_instruction is not None:
            return self._demonstration_generation_instruction
        else:
            self._demonstration_generation_instruction = (
                self.prompt_handler.prompt_dict)["demonstration_generation_instruction"]
            return self._demonstration_generation_instruction

    def init_model(self):
        pass

    def init_config(self):
        pass

    def init_prompt(self):
        pass

    @property
    def prompt_handler(self):
        if self._prompt_handler is not None:
            return self._prompt_handler
        else:
            self._prompt_handler = PromptHandler(class_path=self.FILE_PATH,
                                                 language=self.language,
                                                 class_name=self.class_name,
                                                 prompt_file=self.prompt_file,
                                                 prompt_dict=self.prompt_dict)
            return self._prompt_handler

    @property
    def expand_model_name(self):
        if self.expand_model_config is not None:
            return self.expand_model_config["model_name"]
        else:
            return self.model_name

    def run(self, seed_demonstrations: Union[str, List[str], Dict, Any],
            n: int,
            max_steps: int,
            beam_width: int) -> str:
        self.beam_search_generation(max_steps=max_steps,
                                    beam_width=beam_width,
                                    seed_demonstrations=seed_demonstrations)
        return self.sav_file_path

    def _renew_all_state(self):
        self.all_states = []

    def _update_demonstration_file_name(self, prefix, date, model_name):
        self.sav_file_name = "{}_{}_{}.json".format(prefix, date, model_name)
        self.sav_file_path = os.path.join(self.demonstration_save_dir, self.sav_file_name)

    def _add_demonstrations(self, demonstration: List or dict or str):
        if isinstance(demonstration, List):
            if isinstance(demonstration[0], str):
                self.all_states.extend([json.loads(item) for item in demonstration])
            else:
                self.all_states.extend(demonstration)

        if isinstance(demonstration, dict):
            self.all_states.append(demonstration)

        if isinstance(demonstration, str):
            self.all_states.append(json.loads(demonstration))
        if self.auto_save:
            self._sav_demonstration()

    def _sav_demonstration(self):
        json_file_path = os.path.join(self.demonstration_save_dir, self.sav_file_name)
        logger.info("save self.all_state to pth: {}".format(json_file_path))

        sav_json(data=self.all_states, json_file_path=json_file_path)

    def expand_fn(self, state):
        # Produce some new states from the current state.
        # In a real application, this would involve generating possible next steps.

        """
        Produce some new demonstrations from the current demonstration.
        :param state: list of demonstrations
        # :param expand_config:
        :return expanded: list of lists of demonstrations
        """

        import copy
        # assert ("num_expand" in expand_config.keys()
        #         # and "model_name" in expand_config.keys()
        #         and "demonstration_generation_instruction" in expand_config.keys()
        #         and "demonstration_requirements" in expand_config.keys()), (
        #     "expand_config must contain \"num_expand\", "
        #     # "\"model_name\", "
        #     "\"demonstration_generation_instruction\", "
        #     "\"demonstration_requirements\"")

        expand = []
        # extract the required parameters from expand_config
        # num_expand = expand_config["num_expand"]
        # model_name = expand_config.get("model_name", None)

        # demonstration_generation_instruction = expand_config["demonstration_generation_instruction"]
        logger.info("[demonstration_expand] state: {}".format(state))
        if isinstance(state[-1], str):
            demonstration_text = "\n".join(f"```json\n{item}\n```" for item in state)
        else:
            demonstration_text = "\n".join(f"```json\n{json.dumps(item, ensure_ascii=False)}\n```" for item in state)
            logger.info("demonstration_text: \n{}\n".format(demonstration_text))

        # print("demonstration_text: \n{}\n".format(demonstration_text))
        # demonstration_requirements = expand_config["demonstration_requirements"]
        logger.info("demonstration_requirements: {}".format(self.demonstration_requirements))
        # based on the current demonstration state[-1], expand num_expand demonstrations
        new_demonstrations = self.generate_similar_demonstration(
            demonstration_text=demonstration_text,
            model_name=self.model_name,
            model_config=self.expand_model_config,
            demonstration_requirements=self.demonstration_requirements,
            demonstration_generation_instruction=self.demonstration_generation_instruction,
            num_generated_examples=self.num_expand
        )
        logger.info("new_demonstrations: {}".format(new_demonstrations))
        for new_demonstration in new_demonstrations:
            state_copy = copy.deepcopy(state)
            state_copy.append(new_demonstration)
            expand.append(state_copy)
        return expand

    @staticmethod
    def generate_similar_demonstration(demonstration_text,
                                       demonstration_generation_instruction,
                                       model_name=None,
                                       num_generated_examples=1,
                                       demonstration_requirements=None,
                                       model_config=None):
        """
            generate demonstration based on the reference demonstration (demonstration_text)
            :param demonstration_text:
            :param demonstration_requirements:
            :param model_name:
            :param demonstration_generation_instruction:
            :param num_generated_examples: the number of generated examples
            :param model_config:
            """
        assert (model_config is not None) or (model_name is not None)
        prompt = demo_augmentation_by_llm_prompt_org(
            demonstration_text=demonstration_text,
            demonstration_generation_instruction=demonstration_generation_instruction,
            num_generated_examples=num_generated_examples,
            demonstration_requirements=demonstration_requirements
        )
        if model_config is not None:
            # model_config = convert_model_name_to_model_config(model_config=model_config, add_random=True)
            logger.info(model_config)
            generation_model = GenerationModel(**model_config)
        else:
            model_config = convert_model_name_to_model_config(model_name=model_name, add_random=True)
            generation_model = model_name

        results = call_llm_with_message(prompt, model=generation_model, model_config=model_config)
        logger.info("[generate_similar_demonstration]-generated results: {}".format(results))
        return extract_from_markdown_json(results)

    def score_fn(self, state: list):
        if len(state) == 1:
            return 0
        logger.info("state: {}".format(state))
        logger.info("#query#: \n{}\ntype: {}\n#documents#: \n{}\n".format(state[-1], type(state[-1]), state[:-1]))
        state_copy = [json.dumps(item, ensure_ascii=False) for item in state]
        text_rerank_results = text_rerank(query=state_copy[-1], documents=state_copy[:-1])
        logger.info(text_rerank_results)
        scores = [item["relevance_score"] for item in text_rerank_results["output"]["results"]]
        logger.info(scores)
        if len(scores) == 1:
            logger.info(1 - scores[0])
            return 1 - scores[0]
        else:
            logger.info(1 - np.mean(scores))
            return 1 - np.mean(scores)

    def beam_search_generation(self, max_steps, beam_width, seed_demonstrations):

        # "num_expand" in expand_config.keys()
        # # and "model_name" in expand_config.keys()
        # and "demonstration_generation_instruction" in expand_config.keys()
        # and "demonstration_requirements" in expand_config.keys()

        """
            Performs beam search.

            :param initial_state: List The initial state to begin search from.
            :param max_steps: The maximum number of steps (or iterations) to perform.
            :param beam_width: The number of paths to explore at each step.
            :param expand_fn: A function that takes a state and returns a list of possible next states.
            :param score_fn: A function that takes a state and returns a score. Higher scores are better.
            :param expand_fn_config: the config for the expand_fn, a function that takes a state and returns a list of possible next states.
            :return: The best state found according to the scoring function.
            """

        # initial_state = expand_config["initial_demonstration"]
        initial_state = seed_demonstrations
        logger.info("initial state: {}".format(initial_state))
        logger.info("type of init: {}".format(type(initial_state)))

        self._renew_all_state()
        self._update_demonstration_file_name(prefix="demonstration",
                                             date=get_current_date(),
                                             model_name=self.expand_model_name)
        self._add_demonstrations(initial_state)

        # Initialize the beam with the initial state.
        beam = [[initial_state, self.score_fn(initial_state)]]
        logger.info("beam: {}".format(beam))

        for step in range(max_steps):
            # Expand states in the current beam.
            logger.info("step: {}".format(step))
            candidates = []
            for state, score in beam:
                next_states = self.expand_fn(state)
                logger.info("next_states: \n{}".format(next_states))
                if len(next_states) < 1:
                    logger.info("failed to generate new state: retry!")
                    continue
                for next_state in next_states:
                    self._add_demonstrations(next_state[-1])
                    next_score = self.score_fn(next_state)
                    logger.info("next_score: {}".format(next_score))
                    candidates.append([next_state, next_score])

            # Select the top `beam_width` states for the next iteration.
            logger.info(candidates)
            logger.info("length of candidates: {}".format(len(candidates)))
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_width]

        # Return the state with the highest score after the final iteration.
        best_state, best_score = max(beam, key=lambda x: x[1])
        return best_state, self.all_states
