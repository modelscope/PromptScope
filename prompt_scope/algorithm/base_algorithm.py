from abc import ABC, abstractmethod
from typing import Union

from prompt_scope.core.enumeration.language_enum import LanguageEnum
from prompt_scope.core.utils.prompt_handler import PromptHandler


class PromptOptimizationWithFeedback(ABC):
    """
    Base Abstract Class for Prompt Optimization with Feedback
    """

    def __init__(self,
                 language: LanguageEnum = "cn",
                 **kwargs):
        self.language = language
        self._prompt_handler: Union[PromptHandler, None] = None
        self.kwargs: dict = kwargs

    @property
    def prompt_handler(self):
        """
        Lazily initializes and returns the PromptHandler instance.

        Returns:
            PromptHandler: An instance of PromptHandler initialized with specific file path and keyword arguments.
        """
        if self._prompt_handler is None:
            self._prompt_handler = PromptHandler(self.FILE_PATH, language=self.language, **self.kwargs)
        return self._prompt_handler

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def init_config(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def extract_best_prompt(self):
        pass


class DemonstrationAugmentation(ABC):
    """
    Base Abstract Class for Prompt Optimization with Feedback
    """

    def __init__(self):
        pass

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def init_config(self):
        pass

    @abstractmethod
    def init_prompt(self):
        pass

    @abstractmethod
    def run(self):
        pass
