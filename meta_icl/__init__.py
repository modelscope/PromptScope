
# from . import algorithm
# from .core import enumeration
# from .core import evaluation
# from .core import models
# from .core import offline
# from .core import scheme
# from .core import utils
from .core.utils.registry import Registry

CONFIG_REGISTRY = Registry("config")
PROMPT_REGISTRY = Registry("prompt")

# __all__ = [
#     "algorithm",
#     "enumeration",
#     "evaluation",
#     "models",
#     "offline",
#     "scheme",
#     "utils",
#     "CONFIG_REGISTRY",
#     "PROMPT_REGISTRY",
# ]