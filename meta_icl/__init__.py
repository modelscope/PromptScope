from pathlib import Path
import os

from meta_icl.core.utils.registry import Registry
from meta_icl.core.utils.ipc_config import load_yaml

CONFIG_REGISTRY = Registry("config")
PROMPT_REGISTRY = Registry("prompt")