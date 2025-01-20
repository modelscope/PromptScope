"""
todo: by zy
1. Replace the text rank with other similarity function.
2. Argumentation is conducted in a single round or in a self-instruction manner?
3. max_depth = number of generate samples?
4. Unify the interface of `run` function.
"""
import json
import os
from abc import abstractmethod
from typing import Any
from typing import Union, Dict, List

import numpy as np
from loguru import logger

from prompt_scope.core.enumeration.language_enum import LanguageEnum
from prompt_scope.core.models.generation_model import GenerationModel
from prompt_scope.core.augmentor.demonstration_augmentation.base_demo_augmention import BaseDemonstrationAugmentation
from prompt_scope.core.utils.demontration_utils import demo_augmentation_by_llm_prompt_org
from prompt_scope.core.utils.prompt_handler import PromptHandler
from prompt_scope.core.utils.sys_prompt_utils import (call_llm_with_message, text_rerank,
                                                  convert_model_name_to_model_config)
from prompt_scope.core.utils.sys_prompt_utils import check_dir, sav_json
from prompt_scope.core.utils.utils import extract_from_markdown_json
from prompt_scope.core.utils.utils import get_current_date


class BaseAugmentationByBeamSearch(BaseDemonstrationAugmentation):
    """
    Base class for data augmentation using Beam Search.

    This class extends the BaseDemonstrationAugmentation class and provides a framework
    for data augmentation strategies using abstract methods.
    """

    @abstractmethod
    def expand_fn(self, **kwargs):
        """
        Abstract method for expansion function.

        Defines how to expand the current data. The specific implementation should be provided in subclasses.
        """
        pass

    @abstractmethod
    def score_fn(self, **kwargs):
        """
        Abstract method for scoring function.

        Defines how to score the expanded data. The specific implementation should be provided in subclasses.
        """
        pass

    @abstractmethod
    def beam_search_generation(self, **kwargs):
        """
        Abstract method for Beam Search generation.

        Implements the process of generating augmented data using the Beam Search algorithm.
        This method must be implemented in subclasses.
        """
        pass


class BeamSearchGenerationByDiversity(BaseAugmentationByBeamSearch):
    FILE_PATH: str = __file__
    """
    A class for beam search generation with diversity, extending BaseAugmentationByBeamSearch.
    """
    # Class variable defining the file path for potential file operations or debugging information
    FILE_PATH: str = __file__

    def __init__(self,
                 demonstration_save_dir: int,
                 num_expand: int,
                 demonstration_generation_instruction: str = None,
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
        Initializes the BeamSearchGenerationByDiversity class.
        Args:
            :param demonstration_save_dir (int): Directory path for saving demonstration data.
            :param num_expand (int): Number of expansions in beam search.
            :param demonstration_generation_instruction (str, optional): Instruction for demonstration generation.
            :param language (str, optional): Language used for text processing, default is "cn" (Chinese).
            :param prompt_handler (PromptHandler, optional): An instance of PromptHandler for handling prompt templates.
            :param demonstration_requirements (str, optional): Requirements for demonstration generation.
            :param auto_save (bool, optional): Whether to automatically save the generation results, default is True.
            :param expand_model_config (dict, optional): Configuration for the expansion model.
            :param model_name (str, optional): Name of the model used for generation.
            :param class_name (str, optional): Name of the current class, default is empty.
            :param prompt_file (str, optional): File path for prompt templates.
            :param prompt_dict (dict, optional): A dictionary of prompt templates.
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
        self.class_name = class_name
        self.prompt_file = prompt_file
        self.prompt_dict = prompt_dict
        self._prompt_handler = prompt_handler

    @property
    def demonstration_generation_instruction(self):
        if self._demonstration_generation_instruction is not None:
            return self._demonstration_generation_instruction
        else:
            self._demonstration_generation_instruction = (
                self.prompt_handler.prompt_dict)["Default_Instruction_4_Diverse_Demonstration_Generation"]
            return self._demonstration_generation_instruction

    def init_model(self):
        pass

    def init_config(self):
        pass

    def init_prompt(self):
        pass

    @property
    def prompt_handler(self) -> PromptHandler:
        """
            Retrieves or initializes the PromptHandler instance for this class.

            This property method returns the existing PromptHandler instance if it has been previously initialized.
            If not, it creates a new PromptHandler instance using provided class details
            and assigns it to `_prompt_handler`.

            :return The PromptHandler instance responsible for handling prompts specific to this class.


        """
        if self._prompt_handler is not None:
            return self._prompt_handler
        else:
            logger.info(self.class_name)
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
            max_steps: int,
            beam_width: int,
            n: int = 5) -> str:
        """
        Runs the code generation algorithm based on provided demonstrations.
        This method uses a beam search algorithm to generate code. Given seed demonstrations,
        it searches for the most likely code generation paths within a specified maximum number
        of steps (`max_steps`) and beam width (`beam_width`). It is primarily used in scenarios
        where code is automatically generated, with seed demonstrations being strings, lists of
        strings, dictionaries, or other types of data to guide the generation process.



        :param seed_demonstrations: Union[str, List[str], Dict, Any] Seed demonstrations to guide the code generation process.
        :param max_steps: int Maximum number of steps for code generation, controlling the upper limit of the generated code length.
        :param beam_width: int Beam width for the search, determining how many candidate solutions are retained at each step.
        :param n: int = 5 Number of top generated solutions to retain, defaulting to 5.
        :return The file path where the generated code is saved.

        """
        # todo: by zy, the input parameter `n` is not used.
        #   return the generate results?
        # Execute the beam search code generation process
        self.beam_search_generation(max_steps=max_steps,
                                    beam_width=beam_width,
                                    seed_demonstrations=seed_demonstrations)
        logger.info("demonstration sav path: {}".format(self.sav_file_path))
        return self.all_states

    def _renew_all_state(self) -> None:
        """
        Clear and reinitialize the all_states list.

        This method is used to clear and reinitialize the all_states list within the class lifecycle,
        allowing for a fresh start in recording all states.
        """
        self.all_states = []

    def _update_demonstration_file_name(self, prefix: str, date: str, model_name: str) -> None:
        """
        Update the demonstration file name and path.

        This method generates a new file name by combining the prefix, date, and model name, and sets the file path
        to the corresponding location within the demonstration save directory.

        :param prefix (str): The prefix for the file name.
        :param date (str): The date used in the file name, formatted as 'YYYYMMDD'.
        :param model_name (str): The name of the model, used to distinguish demonstration files from different models.

        """
        self.sav_file_name = "{}_{}_{}.json".format(prefix, date, model_name)
        self.sav_file_path = os.path.join(self.demonstration_save_dir, self.sav_file_name)

    def _add_demonstrations(self, demonstration: List or dict or str):
        """
        Add demonstrations to the dataset.

        This method adds demonstration data to the system's dataset based on the type of the input (list, dictionary, or string).
        For a list, it will add each element (which could be a string or dictionary) to the dataset.
        For a dictionary, it will add the entire dictionary as an entry to the dataset.
        For a string, if auto-save is enabled, it will save the updated dataset after adding the string data.

        :param demonstration: The demonstration data to add, which can be in the form of a list, dictionary, or string.

        """

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
        """
        Save all recorded states during a demonstration.

        This method constructs a JSON file path and serializes all recorded states into JSON format,
        saving them to the specified path.
        It is primarily used to save data after a demonstration for subsequent analysis or reference.

        """
        json_file_path = os.path.join(self.demonstration_save_dir, self.sav_file_name)
        logger.info("save self.all_state to pth: {}".format(json_file_path))

        sav_json(data=self.all_states, json_file_path=json_file_path)

    def expand_fn(self, state: List) -> List:
        """
        Produces new states from the current state.

        In a real application, this would involve generating possible next steps.

        :param state: The current state, represented as a list containing a series of steps or states
        leading to the current point.

        :return A list containing new states generated from the current state.
        """

        import copy
        expand = []
        logger.info("[demonstration_expand] state: {}".format(state))
        # Determine how to handle the state based on the type of the last element in the `state` list
        if isinstance(state[-1], str):
            # If the last element is a string, join the elements in `state` with a specific format
            demonstration_text = "\n".join(f"```json\n{item}\n```" for item in state)
        else:
            # If the last element is not a string,
            # join the elements in `state` with a specific format and use `json.dumps`
            demonstration_text = "\n".join(f"```json\n{json.dumps(item, ensure_ascii=False)}\n```" for item in state)
            logger.info("demonstration_text: \n{}\n".format(demonstration_text))

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
    def generate_similar_demonstration(demonstration_text: Union[str, List[str]],
                                       demonstration_generation_instruction,
                                       model_name=None,
                                       num_generated_examples=1,
                                       demonstration_requirements=None,
                                       model_config=None) -> List:
        """
        Generates similar demonstrations based on the provided text and instructions using a specified model.

        :param demonstration_text: The reference text to mimic in style.
        :param demonstration_generation_instruction: Instructions guiding how the new demonstrations should be generated.
        :param model_name: (optional) The name of the model to use; if not provided, model_config must be given.
        :param num_generated_examples: (optional) The number of examples to generate; defaults to 1.
        :param demonstration_requirements: (optional) Requirements or constraints for the demonstrations.
        :param model_config: (optional) Specific configuration parameters for the model; if not provided, model_name must be given.
        :return: A list of generated demonstration examples.
        """

        # Ensure either model_config or model_name is provided, which is necessary for generating demonstrations
        assert (model_config is not None) or (model_name is not None)

        # Build the initial prompt to generate similar demonstration text via a language model
        prompt = demo_augmentation_by_llm_prompt_org(
            demonstration_text=demonstration_text,
            demonstration_generation_instruction=demonstration_generation_instruction,
            num_generated_examples=num_generated_examples,
            demonstration_requirements=demonstration_requirements
        )

        # Determine how to instantiate the generation model based on whether model_config is provided
        if model_config is not None:
            logger.info(model_config)
            generation_model = GenerationModel(**model_config)
        else:
            # If only model_name is provided, convert it to model configuration
            model_config = convert_model_name_to_model_config(model_name=model_name, add_random=True)
            generation_model = model_name

        # Call the language model with the constructed prompt and model/config to generate demonstration text
        results = call_llm_with_message(prompt, model=generation_model, model_config=model_config)
        logger.info("[generate_similar_demonstration]-generated results: {}".format(results))

        # Extract the required demonstration data from the generated results and return
        return extract_from_markdown_json(results)

    def score_fn(self, state: List) -> float:
        """
        Calculate the score for a given state.

        This function computes a score based on the provided state list, which typically contains a query and multiple documents.
        The function calculates a relevance score based on this information.

        :param state (list): A list containing a query and multiple documents.

        :return The relevance score calculated from the state list.
        """
        if len(state) == 1:
            return 0
        logger.info("state: {}".format(state))
        logger.info("#query#: \n{}\ntype: {}\n#documents#: \n{}\n".format(state[-1], type(state[-1]), state[:-1]))
        state_copy = [json.dumps(item, ensure_ascii=False) for item in state]

        # Call the text_rerank function to rerank documents based on the query and get the results
        # todo: by zy, add similarity function in the future.
        text_rerank_results = text_rerank(query=state_copy[-1], documents=state_copy[:-1])
        logger.info(text_rerank_results)

        # Extract relevance scores from the text_rerank results
        scores = [item["relevance_score"] for item in text_rerank_results["output"]["results"]]
        logger.info(scores)

        # Depending on the length of the scores list, return either the inverse of
        # a single score or the inverse of the mean score
        if len(scores) == 1:
            logger.info(1 - scores[0])
            return 1 - scores[0]
        else:
            logger.info(1 - np.mean(scores))
            return 1 - np.mean(scores)

    def beam_search_generation(self,
                               max_steps: int,
                               beam_width: int,
                               seed_demonstrations: Union[str, List[str]]) -> List:
        """
        Performs beam search generation.

        :param max_steps: The maximum number of steps for the search.
        :param beam_width: The number of candidate sequences to retain at each step.
        :param seed_demonstrations: Initial demonstration sequences to start the search.

        :return best_state: The optimal sequence found after the search.
        :return all_states: All generated sequences.
        """

        # Initialize the search state with the seed demonstrations
        initial_state = seed_demonstrations
        logger.info("initial state: {}".format(initial_state))
        logger.info("type of init: {}".format(type(initial_state)))

        # Renew all internal states
        self._renew_all_state()
        # Update the demonstration file name
        self._update_demonstration_file_name(prefix="demonstration",
                                             date=get_current_date(),
                                             model_name=self.expand_model_name)
        # Add initial state to demonstrations
        self._add_demonstrations(initial_state)

        # Initialize the beam with the initial state.
        beam = [[initial_state, self.score_fn(initial_state)]]
        logger.info("beam: {}".format(beam))

        for step in range(max_steps):
            # Expand states in the current beam.
            logger.info("step: {}".format(step))
            candidates = []
            # Iterate over each state and its score in the current beam
            for state, score in beam:
                # Generate all possible next states using the expand function
                next_states = self.expand_fn(state)
                logger.info("next_states: \n{}".format(next_states))
                if len(next_states) < 1:
                    # If no new states can be generated, log and skip this state
                    logger.info("failed to generate new state: retry!")
                    continue
                for next_state in next_states:
                    # Add the new state to the demonstrations
                    self._add_demonstrations(next_state[-1])
                    # Calculate the score of the new state
                    next_score = self.score_fn(next_state)
                    logger.info("next_score: {}".format(next_score))
                    # Append the new state and its score to the candidate list
                    candidates.append([next_state, next_score])

            logger.info(candidates)
            logger.info("length of candidates: {}".format(len(candidates)))

            # Sort all candidate states by score in descending order
            # and select the top `beam_width` states
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_width]

        # Return the state with the highest score after the final iteration.
        best_state, best_score = max(beam, key=lambda x: x[1])
        return best_state, self.all_states
