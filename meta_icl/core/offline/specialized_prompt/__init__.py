import os
import yaml


config_file_path = os.getenv('CONFIG_FILE_PATH')
LLM_ENV = yaml.safe_load(open(os.path.join(config_file_path, 'ipc_llm_env.yml'), 'r'))