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
r"""The .py version of evaluate_instructions.ipynb to evaluate instructions with a model that can be from the GPT family.

Usage:

Step 1: fill in the instruction(s) you want to evaluate at beginning of main(_)

Step 2: fill in the ratios of training and test splits in your evaluation

Step 3: check if the model configs (like batch size) are the same as the actual serving configs

Step 4: run

```
python evaluate_instructions.py \
    --scorer="text-bison" --dataset="gsm8k" \
    --task="test" --instruction_pos="Q_begin" \
    --evaluate_training_fold=false --evaluate_test_fold=true \
    --openai_api_key="<your_key>" --palm_api_key="<your_key>"
```

The outputs will then be written to `outputs/scorer-outputs/` in the opro folder.
"""

import datetime
import json
import os
import sys

OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, OPRO_ROOT_PATH)

import numpy as np
from opro.evaluation import eval_utils
import pandas as pd

ROOT_DATA_FOLDER_PATH = os.path.join(OPRO_ROOT_PATH, "data")

_SCORER = "qwen2-72b-instruct"

_DATASET = "gsm8k"

_INSTRUCTION_POS = "A_begin"

_EVALUATE_TRAINING_FOLD = False

_EVALUATE_TEST_FOLD = True

_TASK = "test"


def main():
    # set instructions to evaluate
    instructions_to_evaluate = [
        # "",   # 72.5
        "Let's solve the problem.",  # 51.9
        "Let's unify our exceptional intellects and trailblazing methodologies to conquer the adversities and ascend to unexplored heights of unparalleled triumph, harnessing our collective brilliance and avant-garde strategies to surmount any obstacle and achieve superlative success, all while inspiring future generations to embrace our pioneering spirit and relentlessly pursue the pinnacle of human potential.",
        # 48.1
        "Restate the problem in your own words to ensure understanding. Break down the problem into smaller steps, explaining each calculation in detail. Verify each step and re-check your calculations for accuracy. Use proper mathematical notation and maintain consistency with the context of the question.",
        # 31.3
        "Take a deep breath and work on this problem step-by-step.",
    ]
    print(f"instructions_to_evaluate: {instructions_to_evaluate}")

    evaluate_training_fold = _EVALUATE_TRAINING_FOLD
    evaluate_test_fold = _EVALUATE_TEST_FOLD

    assert evaluate_training_fold or evaluate_test_fold
    # set ratios of training and test splits
    train_ratio = 0.0
    test_ratio = 1.0
    assert test_ratio > 0.0 and test_ratio <= 1.0
    if evaluate_training_fold and evaluate_test_fold:
        assert train_ratio + test_ratio == 1

    scorer_llm_name = _SCORER.lower()
    dataset_name = _DATASET.lower()
    task_name = _TASK.lower()
    instruction_pos = _INSTRUCTION_POS

    assert dataset_name in {
        "mmlu",
        "bbh",
        "gsm8k",
        "multiarith",
        "aqua",
    }, (
        "The lower-case dataset name must be one of mmlu, bbh, gsm8k, multiarith,"
        " or aqua."
    )
    if dataset_name == "mmlu":
        assert task_name in {
            "STEM",
            "humanities",
            "social sciences",
            "other (business, health, misc.)",
        }  # for now only support searching on one MMLU category
    elif dataset_name == "bbh":
        assert task_name in {
            "boolean_expressions",
            "causal_judgement",
            "date_understanding",
            "disambiguation_qa",
            "dyck_languages",
            "formal_fallacies",
            "geometric_shapes",
            "hyperbaton",
            "logical_deduction_five_objects",
            "logical_deduction_seven_objects",
            "logical_deduction_three_objects",
            "movie_recommendation",
            "multistep_arithmetic_two",
            "navigate",
            "object_counting",
            "penguins_in_a_table",
            "reasoning_about_colored_objects",
            "ruin_names",
            "salient_translation_error_detection",
            "snarks",
            "sports_understanding",
            "temporal_sequences",
            "tracking_shuffled_objects_five_objects",
            "tracking_shuffled_objects_seven_objects",
            "tracking_shuffled_objects_three_objects",
            "web_of_lies",
            "word_sorting",
        }
    elif dataset_name == "gsm8k":
        assert task_name in {"train", "test"}
    else:
        assert dataset_name in {"multiarith", "aqua"}
        assert task_name == "self"

    assert instruction_pos in {
        "before_Q",
        "Q_begin",
        "Q_end",
        "A_begin",
    }, (
        "The instruction position should be either before the question, or at the"
        " beginning of the question, at the end of the question, or at the"
        " beginning of the answer."
    )

    is_gpt_model = bool(scorer_llm_name in {"gpt-3.5-turbo", "gpt-4"})

    if dataset_name == "mmlu":
        root_data_folder_path = os.path.join(ROOT_DATA_FOLDER_PATH, "MMLU-data")
    elif dataset_name == "bbh":
        root_data_folder_path = os.path.join(
            ROOT_DATA_FOLDER_PATH, "BIG-Bench-Hard-data/"
        )
    elif dataset_name == "gsm8k":
        root_data_folder_path = os.path.join(ROOT_DATA_FOLDER_PATH, "gsm_data")
    elif dataset_name == "aqua":
        root_data_folder_path = os.path.join(ROOT_DATA_FOLDER_PATH, "AQuA-data")
    else:
        assert dataset_name == "multiarith"
        root_data_folder_path = ROOT_DATA_FOLDER_PATH

    # =================== create the result directory ==========================
    datetime_str = (
        str(datetime.datetime.now().replace(microsecond=0))
        .replace(" ", "-")
        .replace(":", "-")
    )
    result_folder = os.path.join(
        OPRO_ROOT_PATH,
        "outputs",
        "scorer-outputs",
        f"{dataset_name.upper()}-{task_name}-s-{scorer_llm_name}-{datetime_str}/",
    )
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    print(f"result directory:\n{result_folder}")

    # ====================== scorer model configs ==============================
    aio_scorer_config = {
        'module_name': 'aio_generation',
        'model_name': scorer_llm_name,
        'max_tokens': 2000,
        'seed': 1234,
        'batch_size': 1,
    }
    scorer_config = {
        'module_name': 'dashscope_generation',
        'model_name': scorer_llm_name,
        'max_tokens': 2000,
        'seed': 1234,
        'batch_size': 1,
    }
    from meta_icl.core.models.generation_model import AioGenerationModel
    import asyncio

    scorer_llm = AioGenerationModel(**aio_scorer_config)
    # ===================== try calling the scorer servers ======================
    print("\n======== testing the scorer server ===========")
    scorer_test_output = [x.message.content for x in asyncio.run(scorer_llm.async_call(
        prompts=["Does the sun rise from the north? Just answer yes or no."])
    )]
    scorer_test_output = [x.message.content for x in asyncio.run(scorer_llm.async_call(
        prompts=["Does the sun rise from the north? Just answer yes or no."])
    )]
    print(f"scorer test output: {scorer_test_output}")
    print("Finished testing the scorer servers.")

    # ====================== read data ============================
    print("\n================ prompt evaluation settings ==============")
    # from https://github.com/hendrycks/test/blob/master/categories.py
    mmlu_subcategories = {
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

    mmlu_categories = {
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
        category_names_to_evaluate = [task_name]
        # one of {'auxiliary_train', 'dev', 'val', 'test'}
        folder_name_to_evaluate = "test"
        task_names_to_evaluate = []
        for task_csv_name in os.listdir(
                os.path.join(root_data_folder_path, folder_name_to_evaluate)
        ):
            task_names_to_evaluate.append(task_csv_name.split(".")[0])

        tasks_in_category = []
        for category_name in category_names_to_evaluate:
            for task_name in task_names_to_evaluate:
                for subname in mmlu_subcategories:
                    if subname in task_name:
                        if mmlu_subcategories[subname][0] in mmlu_categories[category_name]:
                            tasks_in_category.append(task_name)
                            break

        tasks_all = [
            (folder_name_to_evaluate, task_name) for task_name in tasks_in_category
        ]
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
        # # all BBH tasks are as below
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

    elif dataset_name == "gsm8k":
        tasks_all = [task_name]
        multiple_choice_tasks = set()
        boolean_tasks = set()
        numerical_output_tasks = set(tasks_all)
    elif dataset_name == "math":
        tasks_all = [task_name]
        multiple_choice_tasks = set()
        boolean_tasks = set()
        numerical_output_tasks = set()
    elif dataset_name == "aqua":
        tasks_all = [task_name]
        multiple_choice_tasks = set(tasks_all)
        boolean_tasks = set()
        numerical_output_tasks = set()
    else:
        assert dataset_name == "multiarith"
        tasks_all = ["self"]
        multiple_choice_tasks = set()
        boolean_tasks = set()
        numerical_output_tasks = set(tasks_all)

    extract_final_answer_by_prompting_again = False
    include_qa = True
    print(
        f"scorer_llm_name: {scorer_llm_name},"
        " extract_final_answer_by_prompting_again:"
        f" {extract_final_answer_by_prompting_again}, include_qa: {include_qa}\n"
    )
    print("\n================ evaluating instructions ==============")
    print(
        f"dataset: {dataset_name.upper()}, task: {task_name}, instruction_pos:"
        f" {instruction_pos}"
    )

    # ===================== evaluate instructions ==============================
    for t in tasks_all:
        if dataset_name == "mmlu":
            folder_name = t[0]
            task_name = t[1]
            raw_data = pd.DataFrame()
            single_task_df = pd.read_csv(
                os.path.join(root_data_folder_path, f"{folder_name}/{task_name}.csv"),
                index_col=None,
                header=None,
            )
            raw_data = raw_data.append(single_task_df)
            prediction_treat_as_number = False
            prediction_treat_as_bool = False
            num_examples = raw_data.shape[0]
            original_index = np.arange(num_examples)
        elif dataset_name == "bbh":
            task_name = t
            raw_data = []
            single_task_list = eval_utils.load_bbh_task_data(
                task_name, base_dir=root_data_folder_path
            )
            raw_data += single_task_list
            prediction_treat_as_number = bool(
                tasks_all[0] in numerical_output_tasks
            )  # for now only check the first task
            prediction_treat_as_bool = bool(task_name in boolean_tasks)
            num_examples = len(raw_data)
            original_index = np.arange(num_examples)
        elif dataset_name == "gsm8k":
            task_name = t
            raw_data = pd.DataFrame()
            f_gsm = os.path.join(root_data_folder_path, f"gsm_{task_name}.tsv")
            single_task_df = pd.read_csv(f_gsm, sep="\t", header=None)
            raw_data = pd.concat([raw_data, single_task_df])
            prediction_treat_as_number = True
            prediction_treat_as_bool = False
            num_examples = raw_data.shape[0]
            original_index = np.arange(num_examples)
        elif dataset_name == "math":
            task_name = t
            # task_name is already a str; force type conversion to avoid build error
            train_or_test = str(task_name).split("-")[0]
            category_name = str(task_name).split("-")[1]
            with open(
                    os.path.join(
                        root_data_folder_path, train_or_test, f"{category_name}.json"
                    ),
                    "r",
            ) as f:
                raw_data = json.load(f)
            prediction_treat_as_number = "adaptive"
            prediction_treat_as_bool = False
            num_examples = len(raw_data)
            original_index = np.sort(np.array(list(raw_data.keys())).astype(int))
        elif dataset_name == "aqua":
            task_name = t
            raw_data = eval_utils.read_jsonl(
                os.path.join(root_data_folder_path, "AQuA.json")
            )
            prediction_treat_as_number = False
            prediction_treat_as_bool = False
            num_examples = len(raw_data)
            original_index = np.arange(num_examples)
        else:
            assert dataset_name == "multiarith"
            task_name = t
            with open(
                    os.path.join(root_data_folder_path, "MultiArith.json"), "r"
            ) as f:
                raw_data = json.load(f)
            prediction_treat_as_number = True
            prediction_treat_as_bool = False
            num_examples = len(raw_data)
            original_index = np.arange(num_examples)

        is_multiple_choice = bool(task_name in multiple_choice_tasks)
        print(
            f"prediction_treat_as_number: {prediction_treat_as_number},"
            f" prediction_treat_as_bool: {prediction_treat_as_bool},"
            f" is_multiple_choice: {is_multiple_choice}"
        )

        single_task_result_folder = os.path.join(result_folder, task_name)
        os.makedirs(single_task_result_folder)
        scorer_configs_json_path = os.path.join(
            single_task_result_folder, "scorer_configs.json"
        )

        # train-test split
        np.random.seed(0)
        train_index = np.sort(
            np.array(
                np.random.choice(
                    num_examples,
                    size=int(train_ratio * num_examples),
                    replace=False,
                )
            )
        )
        test_index = np.sort(
            np.array(
                np.random.choice(
                    num_examples,
                    size=int(test_ratio * num_examples),
                    replace=False,
                )
            )
        )
        if dataset_name == "math":
            train_index = original_index[train_index]
            test_index = original_index[test_index]
        print(f"total number of exemplars in task: {num_examples}")
        print(
            f"[training fold] whether to evaluate: {evaluate_training_fold},"
            f" number of exemplars: {len(train_index)}"
        )
        print(
            f"[test fold] whether to evaluate: {evaluate_test_fold}, number of"
            f" exemplars: {len(test_index)}"
        )

        for i_ins, instruction in enumerate(instructions_to_evaluate):
            print(
                f"\n({i_ins + 1}/{len(instructions_to_evaluate)}) evaluating"
                f" instruction:\n{instruction}"
            )
            filename = eval_utils.instruction_to_filename(instruction)
            if evaluate_training_fold:
                print("... evaluating the training fold ...")
                detailed_train_results_df = eval_utils.evaluate_single_instruction(
                    data=raw_data,
                    instruction=instruction,
                    eval_index_all=train_index,  # evaluating the training exemplars
                    batch_size=batch_size,
                    call_server_func=call_scorer_server_func,
                    dataset_name=dataset_name,
                    num_servers=num_servers,
                    extract_final_answer_by_prompting_again=extract_final_answer_by_prompting_again,
                    instruction_pos=instruction_pos,
                    is_multiple_choice=is_multiple_choice,
                    include_qa=include_qa,
                    evaluate_in_parallel=evaluate_in_parallel,
                    prediction_treat_as_number=prediction_treat_as_number,
                    prediction_treat_as_bool=prediction_treat_as_bool,
                    prediction_num_decimals=0,
                    verbose=False,
                    max_retry=5,
                    sleep_time=180,
                )
                train_file_path = os.path.join(
                    single_task_result_folder, f"{1 - test_ratio}-TRAIN-{filename}.csv"
                )
                print(f"saving training results to\n{train_file_path}")
                detailed_train_results_df.to_csv(
                    train_file_path, index=True, header=True
                )
                train_scores = detailed_train_results_df["accuracy"]
                print(
                    f"instruction: {instruction}, average training fold accuracy (in"
                    f" percentage): {np.average(train_scores) * 100:.1f}"
                )
            if evaluate_test_fold:
                if scorer_llm_name == 'qwen-max-allinone':
                    semaphore = 30
                else:
                    semaphore = 10
                print("... evaluating the test fold ...")
                detailed_test_results_df = eval_utils.evaluate_single_instruction(
                    data=raw_data,
                    instruction=instruction,
                    eval_index_all=test_index,  # evaluating the test exemplars
                    scorer_llm=scorer_llm,
                    dataset_name=dataset_name,
                    extract_final_answer_by_prompting_again=extract_final_answer_by_prompting_again,
                    instruction_pos=instruction_pos,
                    is_multiple_choice=is_multiple_choice,
                    include_qa=include_qa,
                    evaluate_in_parallel=True,
                    prediction_treat_as_number=prediction_treat_as_number,
                    prediction_treat_as_bool=prediction_treat_as_bool,
                    prediction_num_decimals=0,
                    is_gpt_model=is_gpt_model,
                    verbose=False,
                    max_retry=5,
                    sleep_time=180,
                    semaphore=semaphore,
                )
                test_file_path = os.path.join(
                    single_task_result_folder, f"{test_ratio}-TEST-{filename}.csv"
                )
                print(f"saving test results to\n{test_file_path}")
                detailed_test_results_df.to_csv(test_file_path, index=True, header=True)
                test_scores = detailed_test_results_df["accuracy"]
                print(
                    f"instruction: {instruction}, average test fold accuracy (in"
                    f" percentage): {np.average(test_scores) * 100:.1f}"
                )
            if evaluate_training_fold and evaluate_test_fold:
                print("... concatenating training and test fold results ...")
                detailed_all_results_df = pd.concat(
                    [detailed_train_results_df, detailed_test_results_df]  # pylint: disable=undefined-variable
                )
                detailed_all_results_df = detailed_all_results_df.sort_values(
                    by="index_in_raw_dataset"
                )
                train_and_test_file_path = os.path.join(
                    single_task_result_folder, f"{filename}.csv"
                )
                print(f"saving training + test results to\n{train_and_test_file_path}")
                detailed_all_results_df.to_csv(
                    train_and_test_file_path, index=True, header=True
                )
                all_scores = detailed_all_results_df["accuracy"]
                print(
                    f"instruction: {instruction}, average all fold accuracy (in"
                    f" percentage): {np.average(all_scores) * 100:.1f}"
                )


if __name__ == "__main__":
    main()
