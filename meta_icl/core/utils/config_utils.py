import yaml
import json
from loguru import logger
import argparse


def convert_json_2_yaml(json_file_path, yaml_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
        logger.info(f"load json file: {json_file_path}")
    yaml_text = yaml.dump(data, allow_unicode=True, sort_keys=False)
    print(yaml_text)
    with open(yaml_file_path, 'w') as f:

        f.write(yaml_text)
        logger.info(f"save yaml file to: {yaml_file_path}")



if __name__ == '__main__':

    json_file_path = "conf/app_followup_configs/app_followup_str_conf_BM25.json"
    yaml_file_path = "conf/follwup_configs/general_conf.yaml"
    convert_json_2_yaml(json_file_path, yaml_file_path)
