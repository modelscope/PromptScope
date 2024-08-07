from abc import ABC, abstractmethod

class PromptOptimizationWithFeedback(ABC):
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

    
    