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

from pathlib import Path
from loguru import logger
import asyncio
import hashlib
import json
import os
import re
import string
import time
import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm
from typing import List, Dict, Any

from prompt_scope.datasets import data_loader
from prompt_scope.core.offline.instruction_optimization.opro import metrics
from prompt_scope.core.schemas.message import ChatResponse
from prompt_scope.core.llms.dashscope_llm import DashScopeLlmName
from prompt_scope.core.llms.openai_llm import OpenaiLlmModel


SUBCATEGORIES = {
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

CATEGORIES = {
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


def load_data(*, dataset_name, task_name) -> Dict[str, Any]:
    root_data_folder_path = Path(data_loader.__file__).parent.joinpath(
        'benchmarks', dataset_name)
    if dataset_name == "mmlu":
        # EITHER: filter by category
        # category_names = [
        #     "STEM",
        #     "humanities",
        #     "social sciences",
        #     "other (business, health, misc.)",
        # ]
        category_names = task_name.split(',')
        folder_name = "test"  # one of {'auxiliary_train', 'dev', 'val', 'test'}
        task_names = []
        for task_csv_name in \
        root_data_folder_path.joinpath(folder_name).iterdir():
            task_names.append(task_csv_name.stem)

        print(category_names)
        tasks_in_category = []
        for category_name in category_names:
            for task_name in task_names:
                for subname in SUBCATEGORIES:
                    if subname in task_name:
                        if SUBCATEGORIES[subname][0] in CATEGORIES[category_name]:
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
        assert dataset_name == 'gsm8k'
        tasks_all = [task_name]
        multiple_choice_tasks = set()
        boolean_tasks = set()
        numerical_output_tasks = set(tasks_all)
    
    if dataset_name == "mmlu":
        raw_data = pd.DataFrame()
        prediction_treat_as_number = False
        prediction_treat_as_bool = False
        for t in tasks_all:
            folder_name = t[0]
            task_name = t[1]
            single_task_df = data_loader.MMLUDataLoader(
                    file_path=root_data_folder_path.joinpath(
                        folder_name, f"{task_name}.csv"
                    ),
                    index_col=None,
                    header=None,
                    sep="\t"
                ).load_data()
            raw_data = pd.concat([raw_data, single_task_df])
    
    elif dataset_name == "bbh":
        raw_data = []
        prediction_treat_as_number = bool(
            tasks_all[0] in numerical_output_tasks
        )  # for now only check the first task
        prediction_treat_as_bool = bool(
            tasks_all[0] in boolean_tasks
        )  # for now only check the first task
        for t in tasks_all:
            task_name = t
            raw_data += data_loader.BBHDataLoader(
                    file_path=root_data_folder_path.joinpath(
                        f"{task_name}.json"
                    )
                ).load_data()
        
    elif dataset_name == "gsm8k":
        raw_data = pd.DataFrame()
        prediction_treat_as_number = True
        prediction_treat_as_bool = False
        for t in tasks_all:
            task_name = t
            single_task_df = data_loader.GSMDataLoader(
                    file_path=root_data_folder_path.joinpath(
                        f"gsm_{task_name}.tsv"
                    ),
                    index_col=None,
                    header=None,
                    sep="\t"
                ).load_data()
            raw_data = pd.concat([raw_data, single_task_df])
    num_examples = len(raw_data)
    # ================ split data into train/val/test ==========================
    if dataset_name == "mmlu":
        train_ratio = 0.8
        eval_ratio = 0.2
    elif dataset_name == "gsm8k":
        train_ratio = 0.0035
        eval_ratio = 0
    elif dataset_name == "bbh":
        train_ratio = 0.1
        eval_ratio = 0

    return {"raw_data": raw_data,
            "prediction_treat_as_number": prediction_treat_as_number,
            "prediction_treat_as_bool": prediction_treat_as_bool,
            "multiple_choice_tasks": multiple_choice_tasks,
            "train_ratio": train_ratio,
            "eval_ratio": eval_ratio,
            "num_examples": num_examples}

def remove_punctuation_from_string(input_string, is_filename=True):
    """Remove punctuations from string to comply with filename requirements."""
    # remove punctuations other than "!", "?", "."
    if is_filename:
        punctuation_subset_str = (
            string.punctuation.replace("!", "").replace("?", "").replace(".", "")
        )
        output_string = input_string.translate(
            str.maketrans("", "", punctuation_subset_str)
        )
        # replace punctuations "!", "?", "." with indicating letters
        output_string = (
            output_string.replace("!", "<EXCLAMATION>")
            .replace("?", "<QUESTION>")
            .replace(".", "<PERIOD>")
        )
    else:
        output_string = input_string.translate(
            str.maketrans("", "", string.punctuation)
        )
    return output_string


def instruction_to_filename(instruction, md5_hashing=True):
    """Convert an instruction string to filename."""
    if md5_hashing:
        m = hashlib.md5()
        m.update(instruction.encode("utf-8"))
        filename = m.hexdigest()
    else:
        # remove punctuations and line break, and give a name to the empty string
        filename = instruction.replace("\n", "")
        filename = remove_punctuation_from_string(repr(filename))
        filename = filename if filename else "<NO INSTRUCTION>"
    return filename


def polish_sentence(sentence, add_ending_punc=False):
    """Standardize the sentence to English syntax.

    This is used in prompt optimization to keep track of previously evaluated
    instructions, and is NOT used to create the filename for individual
    instruction results.

    Args:
    sentence (str): the original sentence.
    add_ending_punc (bool): whether to add an ending punctuation.

    Returns:
    sentence (str): the polished sentence.
    """
    sentence = sentence.strip()
    if sentence:
        sentence = sentence.replace("**", "")
        if len(sentence) > 1:
            sentence = (
                    sentence[0].upper() + sentence[1:]
            )  # capitalize the first letter
        if add_ending_punc and not (
                sentence.endswith(".")
                or sentence.endswith("?")
                or sentence.endswith("!")
        ):
            sentence += "."
    return sentence


# pylint: disable=invalid-name
def _split_by_Q(sentence):
    """Split the response and only keep the part before the first "Q:"."""
    return sentence.split("Q:")[0].strip()


def _format_mmlu_example(data, idx, include_question=True):
    """Generate the question part of the MMLU prompt.

    Modified from https://github.com/hendrycks/test/blob/master/evaluate.py.

    Args:
    data (pandas.DataFrame): the comma-delimited MMLU raw data with no index or
        header, and with columns: question, Choice A, Choice B, Choice C, Choice
        D, true answer in ABCD
    idx (int): the index of the question in data
    include_question (bool): whether to include the final question sentence in
        the question. The include_question argument is set to True by default, and
        for now there is no option to change it in gen_prompt.

    Returns:
    prompt (str): the generated question.
    """
    choices = ["(A)", "(B)", "(C)", "(D)"]  # MMLU questions only have 4 choices
    prompt = data.iloc[idx, 0]
    k = data.shape[1] - 2
    for j in range(k):
        prompt += "\n{} {}".format(choices[j], data.iloc[idx, j + 1])
    if include_question:
        prompt += "\nWhat's the answer in (A) (B) (C) (D)?"
    return prompt


def _format_aqua_example(data, idx, include_question=True):
    """Generate the question part of the AQuA prompt."""
    question = data[idx]["question"]
    options = ["(" + item for item in data[idx]["options"]]
    for item in options:
        question += f"\n{item}"
    if include_question:
        question += "\nWhat's the answer in (A) (B) (C) (D) (E)?"
    return question


def gen_prompt(
        data,
        instruction,
        idx,
        include_qa=True,
        instruction_pos="Q_begin",
        dataset_name="mmlu",
):
    """Generate a prompt from the available exemplars and the given instruction.

    The MMLU case was modified from
    https://github.com/hendrycks/test/blob/master/evaluate.py.

    Args:
    data (pandas.DataFrame or list or json): the input-output pairs.
        pandas.DataFrame for MMLU or GSM8K, list for BBH, json for Multiarith.
    instruction (str): the instruction.
    idx (int): the index of the exemplar in the data list.
    include_qa (bool): whether to include "Q:" and "A:" formats in the prompt.
    instruction_pos (str): where to put the instruction, one of {'before_Q',
        'Q_begin', 'Q_end', 'A_begin'}.
    dataset_name (str): one of {"mmlu", "bbh", "gsm8k"}.

    Returns:
    prompt (str): the generated prompt.
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "mmlu":
        question = _format_mmlu_example(data, idx)
    elif dataset_name == "bbh":
        question = data[idx]["input"]
    elif dataset_name == "gsm8k":
        question = data.iloc[idx, 0]
    elif dataset_name == "multiarith":
        question = data[idx]["sQuestion"].strip()
    else:
        assert dataset_name == "aqua"
        question = _format_aqua_example(data, idx)

    prompt = ""
    if include_qa:  # when "Q:" and "A:" are present in the prompt
        if instruction_pos == "before_Q":
            if instruction:
                prompt += instruction + "\n"
                prompt += "Q: " + question
                prompt += "\n\nA:"
        elif instruction_pos == "Q_begin":
            # import pdb;pdb.set_trace()
            if instruction:
                prompt += "Q: " + instruction + "\n"
            else:
                prompt += "Q: "
                prompt += question
                prompt += "\n\nA:"
        elif instruction_pos == "Q_end":
            prompt += "Q: " + question
            if instruction:
                prompt += "\n" + instruction + "\n\nA:"
            else:
                prompt += "\n\nA:"
        else:
            assert instruction_pos == "A_begin"
            prompt += f"Q: {question}\n\n"
            prompt += "A:"
            if instruction:
                prompt += f" {instruction}"
    else:  # when there're no "Q:" and "A:" in the prompt
        assert instruction_pos in {"Q_begin", "Q_end"}
        if instruction_pos == "Q_begin":
            if instruction:
                prompt += instruction + "\n"
                prompt += question
        else:  # instruction_pos == "Q_end"
            prompt += question
            if instruction:
                prompt += "\n" + instruction

    prompt += "The answer can be a sentence, however the final answer should be a number which should be able to correctly answer the problem. Please surround this number with **, e.g. **24**."
    return prompt


def fetch_true_answer(data, idx, dataset_name):
    """Fetch the true answer of the dataset at the idx'th position."""
    dataset_name = dataset_name.lower()
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
        return data.iloc[idx, -1]
    elif dataset_name == "bbh":
        return data[idx]["target"]
    elif dataset_name == "gsm8k":
        return data.iloc[idx, 1]
    elif dataset_name == "multiarith":
        return int(data[idx]["lSolutions"][0])
    else:
        assert dataset_name == "aqua"
        return data[idx]["correct"]


async def async_call_llm(*, llm, list_of_messages, **kwargs) -> List[ChatResponse]:
    # first round of prompting to get raw answers
    tasks = [llm.achat(i, messages, **kwargs) for i, messages in enumerate(list_of_messages)]
    responses = []
    for result in tqdm.as_completed(tasks):
        # As each task completes, the progress bar will be updated
        value = await result
        responses.append(value)
    return [response["result"].message.content for response in sorted(responses, key=lambda x: x["index"])]


def extract_string_in_square_brackets(input_string):
    raw_result = re.findall(r"\[.*?\]", input_string)
    if raw_result:
        return raw_result[0][1:-1]
    else:
        return ""


def parse_tag_content(text, prefix="<TEXT>", suffix="</TEXT>"):
    pattern = f"{prefix}(.*?){suffix}"
    results = re.findall(pattern, text, re.DOTALL)
    return results


def _bucketize_float(num, n_buckets=20):
    assert num >= 0 and num <= 1, "The given number must be between 0 and 1."
    return round(num * n_buckets)


def gen_ins_and_score_pairs_substr(
        old_instructions_and_scores,
        old_instruction_score_threshold=0.1,
        max_num_instructions=1000,
        return_str_only=False,
        num_score_buckets=np.inf,
):
    """Generate the string that includes instruction-score pairs."""
    assert num_score_buckets == np.inf or isinstance(num_score_buckets, int)
    old_instructions_and_scores_str = ""
    old_instructions_and_scores = sorted(
        old_instructions_and_scores, key=lambda x: x[1]
    )[-max_num_instructions:]
    old_instructions_and_scores_in_meta_prompt = []
    for instruction, score, i_step in old_instructions_and_scores:
        if (
                not old_instruction_score_threshold
                or score >= old_instruction_score_threshold
        ):
            old_instructions_and_scores_in_meta_prompt.append(
                (instruction, score, i_step)
            )
            if num_score_buckets == np.inf:
                score_to_show = round(score, 3)
            else:
                score_to_show = _bucketize_float(score, num_score_buckets)
            old_instructions_and_scores_str += (
                f"\ntext:\n{instruction}\nscore:\n{score_to_show}\n"
            )
    if return_str_only:
        return old_instructions_and_scores_str
    else:
        return (
            old_instructions_and_scores_str,
            old_instructions_and_scores_in_meta_prompt,
        )


def gen_meta_prompt(
        old_instructions_and_scores,
        instruction_pos,
        optimizer_llm_name,
        old_instruction_score_threshold=0.1,
        max_num_instructions=1000,
        meta_prompt_type="both_instructions_and_exemplars",
        few_shot_qa_pairs=False,
        include_qa=True,
        data=None,
        few_shot_index_list=None,
        instructions_before_exemplars=True,
        num_score_buckets=np.inf,
        dataset_name="",
        task_name="",
        prompt_handler=None,
):
    """Generate meta prompt for instruction rewriting.

    Args:
    old_instructions_and_scores (list): a list of (instruction, score, i_step)
        pairs.
    instruction_pos (str): where to put the instruction, one of {'before_QA',
        'Q_begin', 'Q_end', 'A_begin'}.
    optimizer_llm_name (str): the name of the LLM used for instruction editing.
    old_instruction_score_threshold (float): only add old instructions with score
        no less than this threshold.
    max_num_instructions (int): the maximum number of instructions in the meta
        prompt.
    meta_prompt_type (str): the type of meta-prompt: whether to have both
        previous instructions and dataset exemplars (often for fine-tuned
        optimizers), or to have only previous instructions (often for pre-trained
        optimizers).
    few_shot_qa_pairs (bool): whether to have few-shot QA pairs in the meta
        prompt.
    include_qa (bool): whether to include "Q:" and "A:" formats in the prompt.
    data (list or pd.DataFrame): the raw data.
    few_shot_index_list (list): the list of indices of few-shot examples.
    instructions_before_exemplars (bool): whether the instruction-score pairs are
        before the exemplars from the dataset.
    num_score_buckets (np.inf or int): the number of score buckets when we
        convert float accuracies to integers. Default to np.inf for not
        bucketizing.
    dataset_name (str): the name of the current dataset. Only used when
        generating task description when meta_prompt_type == "instructions_only".
    task_name (str): the name of the current task. Only used when generating task
        description when meta_prompt_type == "instructions_only".

    Returns:
    meta_prompt (str): the generated meta prompt.
    """
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
    assert meta_prompt_type in {
        "both_instructions_and_exemplars",
        "instructions_only",
    }
    assert dataset_name in {
        "mmlu",
        "bbh",
        "gsm8k",
    }, "The lower-case dataset name must be one of mmlu, bbh, gsm8k."
    assert num_score_buckets == np.inf or isinstance(num_score_buckets, int)

    meta_prompt = ""
    if meta_prompt_type == "both_instructions_and_exemplars":
        if optimizer_llm_name.lower() in [e.value for e in OpenaiLlmModel]:
            if instruction_pos == "A_begin":
                meta_prompt_old_instruction_part = prompt_handler.openai_meta_prompt_old_instruction_part_A_begin
            else:
                meta_prompt_old_instruction_part = prompt_handler.openai_meta_prompt_old_instruction_part_others
        elif optimizer_llm_name.lower() in [e.value for e in DashScopeLlmName]:
            meta_prompt_old_instruction_part = prompt_handler.qwen_meta_prompt_old_instruction_part
            # add old instructions
            old_instructions_and_scores_str = gen_ins_and_score_pairs_substr(
                old_instructions_and_scores=old_instructions_and_scores,
                old_instruction_score_threshold=old_instruction_score_threshold,
                max_num_instructions=max_num_instructions,
                return_str_only=True,
                num_score_buckets=num_score_buckets,
            )
            meta_prompt_old_instruction_part += old_instructions_and_scores_str
        else:
            raise ValueError("Other model type to be implemented")
        # add QA pairs if few_shot_qa_pairs == True
        meta_prompt_exemplar_part = ""
        if few_shot_qa_pairs:
            if optimizer_llm_name.lower() in [e.value for e in OpenaiLlmModel]:
                meta_prompt_exemplar_part += "Below are some problems.\n"
            elif optimizer_llm_name.lower() in [e.value for e in DashScopeLlmName]:
                meta_prompt_exemplar_part += prompt_handler.qwen_meta_prompt_exemplar_part
            else:
                raise ValueError("Other models to be implemented")
        
            for idx in few_shot_index_list:
                if dataset_name == "mmlu":
                    question = _format_mmlu_example(data, idx)  # pylint: disable=protected-access
                    true_answer = data.iloc[idx, -1]
                elif dataset_name == "bbh":
                    question = data[idx]["input"]
                    true_answer = data[idx]["target"]
                else:
                    assert dataset_name == "gsm8k"
                    question = data.iloc[idx, 0]
                    true_answer = data.iloc[idx, 1]

                if include_qa:  # when "Q:" and "A:" are present in the prompt
                    if instruction_pos == "before_Q":
                        meta_prompt_exemplar_part += f"\ninput:\n<INS>\nQ: {question}\nA:"
                    elif instruction_pos == "Q_begin":
                        meta_prompt_exemplar_part += f"\ninput:\nQ: <INS>\n{question}\nA:"
                    elif instruction_pos == "Q_end":
                        meta_prompt_exemplar_part += f"\ninput:\nQ: {question}\n<INS>\nA:"
                    else:  # instruction_pos == "A_begin"
                        if optimizer_llm_name.lower() in [e.value for e in OpenaiLlmModel]:
                            meta_prompt_exemplar_part += f"\nQ: {question}\nA: <Start>"
                        elif optimizer_llm_name.lower() in [e.value for e in DashScopeLlmName]:
                            meta_prompt_exemplar_part += f"\ninput:\nQ: {question}\nA: <INS>"
                        else:
                            raise ValueError("Other models to be implemented")
                else:  # when there're no "Q:" and "A:" in the prompt
                    assert instruction_pos in {"Q_begin", "Q_end"}
                    if optimizer_llm_name.lower() in [e.value for e in OpenaiLlmModel]:
                        if instruction_pos == "Q_begin":
                            meta_prompt_exemplar_part += f"\nProblem:\n<INS>\n{question}\n"
                        elif instruction_pos == "Q_end":
                            meta_prompt_exemplar_part += f"\nProblem:\n{question}\n<INS>\n"
                    elif optimizer_llm_name.lower() in [e.value for e in DashScopeLlmName]:
                        if instruction_pos == "Q_begin":
                            meta_prompt_exemplar_part += f"\ninput:\n<INS>\n{question}\n"
                        elif instruction_pos == "Q_end":
                            meta_prompt_exemplar_part += f"\ninput:\n{question}\n<INS>\n"
                    else:
                        raise ValueError("Other models to be implemented")

                if optimizer_llm_name.lower() in [e.value for e in OpenaiLlmModel]:
                    meta_prompt_exemplar_part += (
                        f"\nGround truth answer:\n{true_answer}\n"
                    )
                elif optimizer_llm_name.lower() in [e.value for e in DashScopeLlmName]:
                    meta_prompt_exemplar_part += f"\noutput:\n{true_answer}\n"
                else:
                    raise ValueError("Other models to be implemented")

        if few_shot_qa_pairs:
            if instructions_before_exemplars:
                meta_prompt += (
                        meta_prompt_old_instruction_part
                        + "\n\n"
                        + meta_prompt_exemplar_part
                )
            else:
                meta_prompt += (
                        meta_prompt_exemplar_part
                        + "\n\n"
                        + meta_prompt_old_instruction_part
                )
        else:
            meta_prompt += meta_prompt_old_instruction_part

        if optimizer_llm_name.lower() in [e.value for e in OpenaiLlmModel]:
            if instruction_pos == "A_begin":
                meta_prompt += prompt_handler.openai_meta_prompt_A_begin
            else:
                meta_prompt += prompt_handler.openai_meta_prompt_others
        elif optimizer_llm_name.lower() in [e.value for e in DashScopeLlmName]:
            meta_prompt += prompt_handler.qwen_meta_prompt
        else:
            raise ValueError("Other models to be implemented")
    else:
        # when using a pre-trained model as optimizer
        assert meta_prompt_type == "instructions_only"

        assert instruction_pos in {"Q_begin", "Q_end", "A_begin"}
        if instruction_pos == "Q_begin":
            instruction_pos_description = "at the beginning of the question"
        elif instruction_pos == "Q_end":
            instruction_pos_description = "at the end of the question"
        else:
            assert instruction_pos == "A_begin"
            instruction_pos_description = "at the beginning of the answer"

        if dataset_name == "gsm8k":
            instruction_task_description = "grade school math"
        elif dataset_name == "mmlu":
            instruction_task_description = task_name
        else:
            assert dataset_name == "bbh"
            instruction_task_description = " ".join(task_name.split("_"))

        meta_instruction = (
            f"Create a piece of text {instruction_pos_description.strip()} to"
            " enhance the precision in solving diverse"
            f" {instruction_task_description.strip()} problems."
        )
        old_instructions_and_scores = sorted(
            old_instructions_and_scores, key=lambda x: x[1]
        )
        old_instructions_and_scores_str = ""
        for instruction, score, _ in old_instructions_and_scores:
            if num_score_buckets == np.inf:
                score_to_show = round(score, 2)
            else:
                score_to_show = _bucketize_float(score, num_score_buckets)
            old_instructions_and_scores_str += (
                f"\n\nPrecision: {score_to_show} <TEXT>{instruction}</TEXT>"
            )
        meta_prompt += meta_instruction + old_instructions_and_scores_str
    return meta_prompt