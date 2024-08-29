# Copyright 2023 The OPRO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""The .py file for prompt optimization.

Usage:

Step 1: edit the starting instructions by modifying `initial_instructions`

Step 2: edit the training ratio by modifying `train_ratio`

Step 3: check if the model configs (like batch size) are the same as the actual serving configs

Step 4: run
"""

import os
import sys
import argparse
import datetime
from pathlib import Path

from meta_icl.core.utils.utils import load_yaml
from loguru import logger
from meta_icl import CONFIG_REGISTRY
from meta_icl.core.offline.instruction_optimization.opro import OPRO
from meta_icl.core.utils.utils import get_current_date

current_file_path = Path(__file__)

logger.add(f"{current_file_path.parent}/log/{current_file_path.stem}_{get_current_date()}.log", rotation="10 MB",
           level="INFO")

WORK_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
import pandas as pd

# ROOT_DATA_FOLDER_PATH = os.path.join(WORK_PATH, "data")
QWEN_MODELS = {"qwen-turbo",
               "qwen2-57b-a14b-instruct",
               "qwen2-72b-instruct",
               "qwen-max",
               "qwen-max-0107",
               }


def config(config_dir=None):
    if config_dir is not None:
        pass
    else:
        logger.info(f"config_dir is None, and loading config from default path")
        try:
            config_dir = os.path.join(os.path.dirname(__file__), "gsm8k_opro.yml")
            logger.info(f"loading config from {config_dir}")
        except Exception as e:
            logger.error(f"error in: {e}")
    args = load_yaml(config_dir)
    return args


def run_gsm_opro(gsm_data_pth):
    scorer_llm_name = CONFIG_REGISTRY.module_dict['model_config'].scorer.model_name
    optimizer_llm_name = CONFIG_REGISTRY.module_dict['model_config'].optim.model_name
    dataset_name = CONFIG_REGISTRY.module_dict['task_config'].dataset_name.lower()
    task_name = CONFIG_REGISTRY.module_dict['basic_config'].task_name

    assert dataset_name == "gsm8k"
    root_data_folder_path = gsm_data_pth

    # =================== create the result directory ==========================
    datetime_str = (
        str(datetime.datetime.now().replace(microsecond=0))
        .replace(" ", "-")
        .replace(":", "-")
    )

    save_folder = os.path.join(
        WORK_PATH,
        CONFIG_REGISTRY.module_dict["basic_config"].output_path,
        "optimization-results",
        f"{dataset_name.upper()}-{task_name}-s-{scorer_llm_name}-o-{optimizer_llm_name}-{datetime_str}/",
    )
    result_by_instruction_folder = os.path.join(
        save_folder, "result_by_instruction"
    )
    os.makedirs(result_by_instruction_folder)
    logger.info(f"result directory:\n{save_folder}")

    # ====================== read data ============================
    logger.info("\n================ prompt optimization settings ==============")
    # from https://github.com/hendrycks/test/blob/master/categories.py

    tasks_all = [task_name]
    multiple_choice_tasks = set()

    raw_data = pd.DataFrame()
    prediction_treat_as_number = True
    prediction_treat_as_bool = False

    for t in tasks_all:
        task_name = t
        f_gsm = os.path.join(root_data_folder_path, f"gsm_{task_name}.tsv")
        single_task_df = pd.read_csv(f_gsm, sep="\t", header=None)
        raw_data = pd.concat([raw_data, single_task_df])

    num_examples = raw_data.shape[0]
    print(f"number of examples in the current task: {num_examples}")

    # ================ split data into train/val/test ==========================
    train_ratio = 0.2
    eval_ratio = 0

    # train-validation-test split
    # It is important to sort the indices, as this ensures the is_multiple_choice
    # Boolean variables match the data points.
    assert train_ratio + eval_ratio <= 1
    test_ratio = 1 - train_ratio - eval_ratio
    print(
        f"train_ratio: {train_ratio}, eval_ratio: {eval_ratio}, "
        f"test_ratio: {test_ratio}"
    )
    np.random.seed(0)
    train_index = np.sort(
        np.array(
            np.random.choice(
                num_examples, size=int(train_ratio * num_examples), replace=False
            )
        )
    )
    eval_and_test_index = np.sort(
        np.array(list(set(np.arange(num_examples)) - set(train_index)))
    )
    eval_index = np.sort(
        np.array(
            np.random.choice(
                eval_and_test_index,
                size=int(eval_ratio * num_examples),
                replace=False,
            )
        )
    )

    few_shot_selection_criteria = "random"

    # ===================== run prompt optimization ======================

    assert few_shot_selection_criteria in {
        "accumulative_most_frequent",
        "current_most_frequent",
        "random",
        "constant",
    }
    additional_kwargs = {
        "tasks_all": tasks_all,
        "train_ratio": train_ratio,
        "eval_ratio": eval_ratio,
        "test_ratio": test_ratio,
        "train_index": train_index,
        "eval_index": eval_index,
        "num_examples": num_examples,
        "root_data_folder_path": root_data_folder_path,
        "multiple_choice_tasks": multiple_choice_tasks,
        "raw_data": raw_data,
        "prediction_treat_as_number": prediction_treat_as_number,
        "prediction_treat_as_bool": prediction_treat_as_bool,
        "result_by_instruction_folder": result_by_instruction_folder,
        "save_folder": save_folder,
    }

    pipeline = OPRO(language="en")
    pipeline.update_config(**additional_kwargs)
    pipeline.run()


if __name__ == '__main__':
    config_dir = "examples/gsm8k_example/configs/gsm8k_opro.yml"
    gsm_data_pth = "examples/gsm8k_example/data/gsm_data_showcase"

    args = config(config_dir=config_dir)
    logger.info(args)
    CONFIG_REGISTRY.batch_register(args)
    logger.info(CONFIG_REGISTRY.module_dict)
    run_gsm_opro(gsm_data_pth)
