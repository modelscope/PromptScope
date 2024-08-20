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

from meta_icl.core.utils.utils import load_yaml
from meta_icl.core.utils.logger import Logger
from meta_icl import CONFIG_REGISTRY
from meta_icl.core.offline.instruction_optimization.opro import OPRO

WORK_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
from meta_icl.algorithm.opro.optimization import opt_utils
import pandas as pd

ROOT_DATA_FOLDER_PATH = os.path.join(WORK_PATH, "data")
QWEN_MODELS = {"qwen-turbo",
			   "qwen2-57b-a14b-instruct",
			   "qwen2-72b-instruct",
			   "qwen-max",
			   "qwen-max-0107",
			   }

def config():
	config_dir = os.path.join(os.path.dirname(__file__), "opro.yml")
	args = load_yaml(config_dir)
	return args

def main():
	scorer_llm_name = CONFIG_REGISTRY.module_dict['model_config'].scorer.model_name
	optimizer_llm_name = CONFIG_REGISTRY.module_dict['model_config'].optim.model_name
	dataset_name = CONFIG_REGISTRY.module_dict['task_config'].dataset_name.lower()
	task_name = CONFIG_REGISTRY.module_dict['basic_config'].task_name

	if dataset_name == "mmlu":
		root_data_folder_path = os.path.join(ROOT_DATA_FOLDER_PATH, "MMLU-data")
	elif dataset_name == "bbh":
		root_data_folder_path = os.path.join(
			ROOT_DATA_FOLDER_PATH, "BIG-Bench-Hard-data/"
		)
	else:
		assert dataset_name == "gsm8k"
		root_data_folder_path = os.path.join(ROOT_DATA_FOLDER_PATH, "gsm_data")

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
	print(f"result directory:\n{save_folder}")

  # ====================== read data ============================
	print("\n================ prompt optimization settings ==============")
	# from https://github.com/hendrycks/test/blob/master/categories.py
	subcategories = {
		"abstract_algebra": ["math"],
		"anatomy": ["health"],
		"astronomy": ["physics"],
		"business_ethics": ["business"],
		"clinical_knowledge": ["health"],
		"college_biology": ["biology"],
		"college_chemistry": ["chemistry"],
		"college_computer_science": ["computer science"],
		"college_mathematics": ["math"],
		"college_medicine": ["health"],
		"college_physics": ["physics"],
		"computer_security": ["computer science"],
		"conceptual_physics": ["physics"],
		"econometrics": ["economics"],
		"electrical_engineering": ["engineering"],
		"elementary_mathematics": ["math"],
		"formal_logic": ["philosophy"],
		"global_facts": ["other"],
		"high_school_biology": ["biology"],
		"high_school_chemistry": ["chemistry"],
		"high_school_computer_science": ["computer science"],
		"high_school_european_history": ["history"],
		"high_school_geography": ["geography"],
		"high_school_government_and_politics": ["politics"],
		"high_school_macroeconomics": ["economics"],
		"high_school_mathematics": ["math"],
		"high_school_microeconomics": ["economics"],
		"high_school_physics": ["physics"],
		"high_school_psychology": ["psychology"],
		"high_school_statistics": ["math"],
		"high_school_us_history": ["history"],
		"high_school_world_history": ["history"],
		"human_aging": ["health"],
		"human_sexuality": ["culture"],
		"international_law": ["law"],
		"jurisprudence": ["law"],
		"logical_fallacies": ["philosophy"],
		"machine_learning": ["computer science"],
		"management": ["business"],
		"marketing": ["business"],
		"medical_genetics": ["health"],
		"miscellaneous": ["other"],
		"moral_disputes": ["philosophy"],
		"moral_scenarios": ["philosophy"],
		"nutrition": ["health"],
		"philosophy": ["philosophy"],
		"prehistory": ["history"],
		"professional_accounting": ["other"],
		"professional_law": ["law"],
		"professional_medicine": ["health"],
		"professional_psychology": ["psychology"],
		"public_relations": ["politics"],
		"security_studies": ["politics"],
		"sociology": ["culture"],
		"us_foreign_policy": ["politics"],
		"virology": ["health"],
		"world_religions": ["philosophy"],
	}

	categories = {
		"STEM": [
			"physics",
			"chemistry",
			"biology",
			"computer science",
			"math",
			"engineering",
		],
		"humanities": ["history", "philosophy", "law"],
		"social sciences": [
			"politics",
			"culture",
			"economics",
			"geography",
			"psychology",
		],
		"other (business, health, misc.)": ["other", "business", "health"],
	}

	if dataset_name == "mmlu":
		# EITHER: filter by category
		# category_names = [
		#     "STEM",
		#     "humanities",
		#     "social sciences",
		#     "other (business, health, misc.)",
		# ]
		category_names = [task_name]
		folder_name = "test"  # one of {'auxiliary_train', 'dev', 'val', 'test'}
		task_names = []
		for task_csv_name in os.listdir(
			os.path.join(root_data_folder_path, folder_name)
		):
			task_names.append(task_csv_name.split(".")[0])

		tasks_in_category = []
		for category_name in category_names:
			for task_name in task_names:
				for subname in subcategories:
					if subname in task_name:
						if subcategories[subname][0] in categories[category_name]:
							tasks_in_category.append(task_name)
							break

		tasks_all = [(folder_name, task_name) for task_name in tasks_in_category]
		multiple_choice_tasks = set([item[1] for item in tasks_all])
		boolean_tasks = set()
		numerical_output_tasks = set()

		# OR: filter by task
		# tasks_all = [
		#     # ('test', 'abstract_algebra_test'),
		#     # ('test', 'college_computer_science_test'),
		#     # ('test', 'college_mathematics_test'),
		#     # ('test', 'college_physics_test'),
		#     # ('test', 'elementary_mathematics_test'),
		#     # ('test', 'global_facts_test'),
		#     # ('test', 'high_school_physics_test'),
		#     # ('test', 'machine_learning_test'),
		#     # ('test', 'management_test'),
		#     # ('test', 'medical_genetics_test'),
		#     # ('test', 'moral_scenarios_test'),
		#     # ('test', 'professional_psychology_test'),
		#     # ('test', 'public_relations_test'),
		#     # ('test', 'professional_law_test'),
		#     # ('test', 'high_school_psychology_test'),
		#     # ('test', 'high_school_world_history_test'),
		#     # ('test', 'human_aging_test'),
		#     # ('test', 'miscellaneous_test'),
		#     # ('test', 'moral_scenarios_test'),
		#     ('test', 'professional_psychology_test'),
		#     # ('test', 'security_studies_test'),
		# ]

	elif dataset_name == "bbh":
		tasks_all = [task_name]
		assert (
			len(tasks_all) == 1
		), "for now only support prompt optimization on one BBH task"

		# all BBH tasks are as below
		# tasks_all = [
		#     'boolean_expressions',
		#     'causal_judgement',
		#     'date_understanding',
		#     'disambiguation_qa',
		#     'dyck_languages',
		#     'formal_fallacies',
		#     'geometric_shapes',
		#     'hyperbaton',
		#     'logical_deduction_five_objects',
		#     'logical_deduction_seven_objects',
		#     'logical_deduction_three_objects',
		#     'movie_recommendation',
		#     'multistep_arithmetic_two',
		#     'navigate',
		#     'object_counting',
		#     'penguins_in_a_table',
		#     'reasoning_about_colored_objects',
		#     'ruin_names',
		#     'salient_translation_error_detection',
		#     'snarks',
		#     'sports_understanding',
		#     'temporal_sequences',
		#     'tracking_shuffled_objects_five_objects',
		#     'tracking_shuffled_objects_seven_objects',
		#     'tracking_shuffled_objects_three_objects',
		#     'web_of_lies',
		#     'word_sorting'
		# ]
		numerical_output_tasks = {
			"object_counting",
			"multistep_arithmetic_two",
		}

		multiple_choice_tasks = {
			"date_understanding",
			"disambiguation_qa",
			"geometric_shapes",
			"hyperbaton",
			"logical_deduction_five_objects",
			"logical_deduction_seven_objects",
			"logical_deduction_three_objects",
			"movie_recommendation",
			"penguins_in_a_table",
			"reasoning_about_colored_objects",
			"ruin_names",
			"salient_translation_error_detection",
			"snarks",
			"temporal_sequences",
			"tracking_shuffled_objects_five_objects",
			"tracking_shuffled_objects_seven_objects",
			"tracking_shuffled_objects_three_objects",
		}

		boolean_tasks = {
			"boolean_expressions",  # True or False
			"causal_judgement",  # yes or no
			"formal_fallacies",  # valid or invalid
			"navigate",  # yes or no
			"sports_understanding",  # yes or no
			"web_of_lies",  # yes or no
		}

	else:
		assert dataset_name in {"gsm8k"}
		tasks_all = [task_name]
		multiple_choice_tasks = set()
		boolean_tasks = set()
		numerical_output_tasks = set(tasks_all)

	if dataset_name == "mmlu":
		raw_data = pd.DataFrame()
		prediction_treat_as_number = False
		prediction_treat_as_bool = False
	elif dataset_name == "bbh":
		raw_data = []
		prediction_treat_as_number = bool(
			tasks_all[0] in numerical_output_tasks
		)  # for now only check the first task
		prediction_treat_as_bool = bool(
			tasks_all[0] in boolean_tasks
		)  # for now only check the first task
		print(
			f"prediction_treat_as_number: {prediction_treat_as_number},"
			f" prediction_treat_as_bool: {prediction_treat_as_bool}"
		)
	else:
		assert dataset_name == "gsm8k"
		raw_data = pd.DataFrame()
		prediction_treat_as_number = True
		prediction_treat_as_bool = False

	for t in tasks_all:
		if dataset_name == "mmlu":
			folder_name = t[0]
			task_name = t[1]
			single_task_df = pd.read_csv(
				os.path.join(root_data_folder_path, f"{folder_name}/{task_name}.csv"),
				index_col=None,
				header=None,
			)
			raw_data = pd.concat([raw_data, single_task_df])
		elif dataset_name == "bbh":
			task_name = t
			single_task_list = opt_utils.load_bbh_task_data(
				task_name, base_dir=root_data_folder_path
			)
			raw_data += single_task_list
		else:
			assert dataset_name == "gsm8k"
			task_name = t
			f_gsm = os.path.join(root_data_folder_path, f"gsm_{task_name}.tsv")
			single_task_df = pd.read_csv(f_gsm, sep="\t", header=None)
			raw_data = pd.concat([raw_data, single_task_df])

	if dataset_name == "mmlu":
		num_examples = raw_data.shape[0]
	elif dataset_name == "bbh":
		num_examples = len(raw_data)
	else:
		assert dataset_name in {"gsm8k"}
		num_examples = raw_data.shape[0]
	print(f"number of examples in the current task: {num_examples}")

  # ================ split data into train/val/test ==========================
	if dataset_name == "mmlu":
		train_ratio = 0.8
		eval_ratio = 0.2
	elif dataset_name == "gsm8k":
		train_ratio = 0.0035
		eval_ratio = 0
	else:
		assert dataset_name == "bbh"
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
	# logger = Logger.get_logger(__name__)
	args = config()
	# logger.info(args)
	CONFIG_REGISTRY.batch_register(args)
	print(CONFIG_REGISTRY.module_dict)
	main()
