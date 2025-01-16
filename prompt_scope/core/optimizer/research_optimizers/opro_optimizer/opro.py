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
import collections
import os
import re
import pickle as pkl
import numpy as np
import pandas as pd
from pydantic import Field
from typing import List, Literal, Dict, Set, Sequence, Any, Tuple
from loguru import logger
from datetime import datetime
import json
from pathlib import Path
import asyncio

from prompt_scope.core.optimizer.research_optimizers.base_algorithm import PromptOptimizationWithFeedback
from prompt_scope.core.evals.loading import load_evaluator
from prompt_scope.core.evals.schema import StringEvaluator
from prompt_scope.core.llms.base import BaseLLM
from prompt_scope.core.llms.openai_llm import OpenaiLLM, OpenaiLlmModel
from prompt_scope.core.llms.dashscope_llm import DashscopeLLM, DashScopeLlmName
from prompt_scope.core.optimizer.research_optimizers.opro_optimizer import metrics
from prompt_scope.core.optimizer.research_optimizers.opro_optimizer.utils import (
    load_data, 
    fetch_true_answer, 
    gen_prompt, 
    async_call_llm, 
    _split_by_Q, 
    instruction_to_filename, 
    gen_ins_and_score_pairs_substr,
    gen_meta_prompt,
    extract_string_in_square_brackets,
    polish_sentence,
    parse_tag_content,
    )
from prompt_scope.datasets import data_loader


def create_dated_directory(base_path):
    # Get current date in YYYYMMDD format
    current_date = datetime.now().strftime("%Y%m%d")
    version = 1
    
    # Keep incrementing version until we find a directory that doesn't exist
    while True:
        dir_name = f"{current_date}_v{version}"
        full_path = os.path.join(base_path, dir_name)
        
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            return full_path
        
        version += 1

class OPRO(PromptOptimizationWithFeedback):
    """
	OPRO (Optimization by PROmpting) is a system designed to iteratively refine and optimize
	prompts for language models (Large Language Models as Optimizers). It manages configurations, initializes models, validates parameters,
	and orchestrates a step-by-step process to create, assess, and refine instructions or prompts
	targeting specific tasks within designated datasets such as MMLU, BBH, or GSM8K. Leveraging Qwen models
	for scoring and optimization strategies, it supports both instruction-centric and example-based
	meta-prompts, maintaining a record of progress through comprehensive evaluations to incrementally
	enhance the efficacy of generated instructions.
	"""
    # =============LLM Configuration=============
    scorer_llm: BaseLLM = Field(default=DashscopeLLM())
    optim_llm: BaseLLM = Field(default=DashscopeLLM())
    scorer_llm_name: str = Field(default=DashScopeLlmName.QWEN2_7B_INST)
    optim_llm_name: str = Field(default=DashScopeLlmName.QWEN2_7B_INST)

    # =============Path Configuration=============
    dataset_name: Literal['mmlu', 'gsm8k', 'bbh', 'thunews', 'cmmlu'] = Field(..., description="Name of the dataset")
    task_name: str | List[str] = Field(...)
    prompt_path: str = Field(default=__file__, description="Prompt file path")
    store_path: str = Field(default=Path(__file__).parent.joinpath("opro_output"))

    # =============Experiment Configuration=============
    meta_prompt_type: Literal["both_instructions_and_exemplars", "instructions_only"] = Field(...)
    instruction_pos: Literal["before_QA", "Q_begin", "Q_end", "A_begin"] = Field(...)
    optimizer_llm_temperature_schedule: Literal["constant", "linear_increase"] | None = None
    optimizer_llm_temperature: float = 0.8
    optimizer_llm_temperature_end: float | None = None
    train_ratio: float | None = None
    eval_ratio: float | None = None
    train_index: Sequence | None = None
    eval_index: Sequence | None = None
    
    old_instruction_score_threshold: float = 0.0
    extract_final_answer_by_prompting_again: bool = False
    include_qa: bool = False
    evaluate_in_parallel: bool = True
    generate_in_parallel: bool = True
    # instructions: List[str] = ["Please break down the problem into smaller, manageable steps and solve it systematically. After solving each part, verify your calculations to ensure accuracy, and explain your reasoning for each step. Finally, clearly state the final answer by surrounding it with **, like this: \\(**final answer**). For example, if the final answer is 16, you should write: 'Thus, the final answer is \\(**16**)."]
    # instructions: List[str] = ["Let me help you count the items you have. Just list them one by one, separated by commas. I will then count each item and tell you how many items there are in total."]
    # instructions: List[str] = ["Please think step by step.", "Please follow the instructions with meticulous attention and sort the provided list of words into perfect alphabetical order. Ensure that each word is precisely placed in its correct position according to the alphabetical sequence, exactly as demonstrated in the given examples. Carefully verify that your final output matches the specified format without any discrepancies. Adhere to each step with utmost precision and diligence to guarantee the highest accuracy."]
    instructions: List[str] = ["请选择正确的选项来回答下列问题"]
    few_shot_qa_pairs: bool = True
    num_score_buckets: int = 100
    max_num_instructions: int = 20
    meta_prompt_instructions_before_exemplars: bool = True
    few_shot_selection_criteria: Literal["accumulative_most_frequent",
        "current_most_frequent",
        "random",
        "constant"] = "constant"
    num_generated_instructions_in_each_step: int = 8
    evaluate_generated_ins_on_few_shot: bool = False
    num_few_shot_questions_for_instruction_refinement: int = 5
    evaluate_old_ins_on_few_shot: bool = False
    eval_interval: int = 3
    verbose: bool = False
    batch_size: int = 1
    semaphore: int = 10
    prediction_num_decimals: int = 0
    
    
    def evaluate_single_instruction(self,
                                    *,
                                    instruction: str,
                                    raw_data: Sequence,
                                    index_to_evaluate: Sequence,
                                    true_answers: Sequence,
                                    prediction_treat_as_number: bool | Literal["adaptive"],
                                    prediction_treat_as_bool: bool,
                                    prediction_treat_as_text: bool,
                                    is_multiple_choice: bool | List[bool],
                                    ) -> pd.DataFrame:
        prediction_metadata = self._predict(
            instruction=instruction, 
            raw_data = raw_data,
            eval_index_all=index_to_evaluate
            )
        if self.extract_final_answer_by_prompting_again:
            raw_answers = prediction_metadata['raw_answers_second_round']
        else:
            raw_answers = prediction_metadata['raw_answers']
        
        raw_prompts_flattened = prediction_metadata['raw_prompts_flattened']
        choices, accuracies = self._evaluate_and_analyze(
            raw_answers=raw_answers, 
            true_answers=true_answers,
            raw_prompts_flattened=raw_prompts_flattened,
            instruction=instruction,
            prediction_treat_as_number=prediction_treat_as_number,
            prediction_treat_as_bool=prediction_treat_as_bool,
            prediction_treat_as_text=prediction_treat_as_text,
            is_multiple_choice_all=is_multiple_choice,
            )
        detailed_results_df = self.metadata_to_df(
            eval_index_all=index_to_evaluate,
            choices=choices,
            true_answers=true_answers,
            accuracies=accuracies, 
            **prediction_metadata, 
            )
        return detailed_results_df
    def _before_run(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        data_dict = load_data(dataset_name=self.dataset_name, task_name=self.task_name)
        train_ratio = self.train_ratio or data_dict['train_ratio']
        eval_ratio = self.eval_ratio or data_dict['eval_ratio']
        data_dict['train_ratio'] = train_ratio
        data_dict['eval_ratio'] = eval_ratio
        num_examples = data_dict['num_examples']
        raw_data = data_dict['raw_data']
        prediction_treat_as_number = data_dict['prediction_treat_as_number']
        prediction_treat_as_bool = data_dict['prediction_treat_as_bool']
        prediction_treat_as_text = data_dict.get('prediction_treat_as_text', False)
        
        assert train_ratio + eval_ratio <= 1
        test_ratio = 1 - train_ratio - eval_ratio

        np.random.seed(0)
        # train_index = self.train_index or np.sort(
        #     np.array(
        #         np.random.choice(
        #             num_examples, size=int(train_ratio * num_examples), replace=False
        #         )
        #     )
        # )
        # eval_and_test_index = np.sort(
        #     np.array(list(set(np.arange(num_examples)) - set(train_index)))
        # )
        # eval_index = self.eval_index or np.sort(
        #     np.array(
        #         np.random.choice(
        #             eval_and_test_index,
        #             size=int(eval_ratio * num_examples),
        #             replace=False,
        #         )
        #     )
        # )

        train_index = self.train_index or np.sort(
            np.array(
                np.random.choice(
                    num_examples, size=int(train_ratio * num_examples), replace=False
                )
            )
        )
        eval_and_test_index = np.sort(
            np.array(list(set(np.arange(num_examples)) - set(train_index)))
        )
        eval_index = self.eval_index or np.sort(
            np.array(
                np.random.choice(
                    eval_and_test_index,
                    size=int(eval_ratio * num_examples),
                    replace=False,
                )
            )
        )
        
        if self.dataset_name == "mmlu":
            is_multiple_choice = True
            is_multiple_choice_eval = True
        elif self.dataset_name == "gsm8k":
            is_multiple_choice = False
            is_multiple_choice_eval = False
        elif self.dataset_name == 'bbh':
            is_multiple_choice = []
            is_multiple_choice_eval = []
            multiple_choice_tasks = data_dict["multiple_choice_tasks"]
            train_index_by_task_dict = dict()
            eval_index_by_task_dict = dict()
            start_index = 0
            tasks_all = [self.task_name]
            root_data_folder_path = Path(data_loader.__file__).parent.joinpath('benchmarks', self.dataset_name)
            for task_name in tasks_all:
                single_task_list = data_loader.BBHDataLoader(
                    file_path=root_data_folder_path.joinpath(
                        f"{task_name}.json"
                    )
                ).load_data()
                end_index = start_index + len(single_task_list)
                train_index_by_task_dict[task_name] = (
                    train_index[(train_index >= start_index) & (train_index < end_index)]
                    # if " - start_index" is added here, then the dict would contain
                    # indices in the original task
                )
                eval_index_by_task_dict[task_name] = (
                    eval_index[(eval_index >= start_index) & (eval_index < end_index)]
                    # if " - start_index" is added here, then the dict would contain
                    # indices in the original task
                )
                start_index = end_index
                is_multiple_choice_single_task_train = [
                                                            task_name in multiple_choice_tasks
                                                        ] * len(train_index_by_task_dict[task_name])
                is_multiple_choice_single_task_eval = [
                                                            task_name in multiple_choice_tasks
                                                        ] * len(eval_index_by_task_dict[task_name])
                is_multiple_choice += is_multiple_choice_single_task_train
                is_multiple_choice_eval += is_multiple_choice_single_task_eval
        elif self.dataset_name in ['thunews', 'cmmlu']:
            is_multiple_choice = False
            is_multiple_choice_eval = False
            # self.train_index = list(range(200))
            # self.eval_index = list(range(200, 300))

        if not self.optimizer_llm_temperature_end:
            self.optimizer_llm_temperature_schedule = None

        logger.info(
            f"train_ratio: {train_ratio}, number of training points:"
            f" {len(train_index)}"
        )
        logger.info(
            f"eval_ratio: {eval_ratio}, number of eval points: "
            f"{len(eval_index)}"
        )
        logger.info(
            f"test_ratio: {test_ratio}, number of test points: "
            f"{int(num_examples * test_ratio)}"
        )
        
        logger.info(f"task_name: {self.task_name}")
        logger.info(
            f"generating {self.num_generated_instructions_in_each_step} instructions in"
            f" each step, run for {self.num_steps} steps"
        )
        logger.info(
            "discarding generated instructions with score less than:"
            f" {self.old_instruction_score_threshold} (old_instruction_score_threshold)"
        )
        logger.info(f"num_score_buckets: {self.num_score_buckets}")

        # =================== save configurations to json file ====================
        configs = {k: v for k, v in data_dict.items() if k not in ['raw_data', 'multiple_choice_tasks']}
        logger.info(configs)
        self.store_path = create_dated_directory(self.store_path)
        with open(os.path.join(self.store_path, "configs_dict.json"), "w") as f:
            json.dump(configs, f, indent=4)

        detailed_results_df_by_instruction_dict = dict()
        old_instructions_and_scores = []
        old_instructions_and_scores_raw = []
        instruction_score_dict = dict()
        wrong_questions_from_start_counter = collections.Counter()
        
        # evaluate initial instructions
        logger.info("\n============== evaluating initial instructions on train index ===============")
        true_answers = [
                str(fetch_true_answer(raw_data, idx=idx, dataset_name=self.dataset_name))
                for idx in train_index]
        for instruction in self.instructions:
            detailed_results_df = self.evaluate_single_instruction(
                instruction=instruction, 
                raw_data=raw_data,
                index_to_evaluate=train_index,
                true_answers=true_answers,
                prediction_treat_as_number=prediction_treat_as_number,
                prediction_treat_as_bool=prediction_treat_as_bool,
                prediction_treat_as_text=prediction_treat_as_text,
                is_multiple_choice=is_multiple_choice,
            )
            # accuracies, **prediction_metadata,
            detailed_results_df_by_instruction_dict[instruction] = detailed_results_df
            scores = detailed_results_df["accuracy"]
            average_score = np.average(scores)
            logger.info(f"instruction: {instruction}, score: {average_score}")
        filename = instruction_to_filename(instruction)
        file_path = os.path.join(self.store_path, f"{filename}.csv")
        detailed_results_df.to_csv(file_path, index=True, header=True)
        logger.info(f"""saving results of "{instruction}" to {file_path}""")


        # prepare result-related keyword arguments
        old_instructions_and_scores.append((instruction, average_score, -1))
        old_instructions_and_scores_raw.append((instruction, average_score, -1))
        instruction_score_dict[instruction] = average_score
        # increment the counter on wrong questions
        wrong_question_indices_set = set(
            list(
                detailed_results_df.iloc[
                np.where(detailed_results_df.accuracy == 0.0)[0], :
                ].index
            )
        )
        for idx in wrong_question_indices_set:
            wrong_questions_from_start_counter[idx] += 1
        
        result_kwargs = {
            # key: step index; value: the list of few-shot indices in that step"
            # the dictionary of the few-shot QA indices in meta-prompt.
            "few_shot_index_list_by_step_dict": dict(), 
            "generated_ins_on_few_shot_results_dict": dict(),
            "old_ins_on_few_shot_results_dict": dict(),
            # evaluation results every a few steps
            # format: [(i_step, instruction, detailed_results_df)]
            "eval_results": [],
            "eval_detailed_results_df_by_instruction_dict": dict(),  # {instruction: detailed_results_df}
            "instruction_eval_score_dict": dict(),  # {instruction: eval_score}
            "old_instruction_md5_hashstrings_set": set(),
            "prev_saved_instructions": set(),
            # all generated instructions, format: [(instruction, score, step_index)]
            # the instructions that were skipped have score NaN
            "old_instructions_and_scores_raw": old_instructions_and_scores_raw,
            # the new instructions, format: [(instruction, score, step_index)]
            "old_instructions_and_scores": old_instructions_and_scores,
            "meta_prompts": [],  # format: [(meta_prompt, step_index)]
            "instruction_score_dict": instruction_score_dict, # the dictionary of {instruction: score}
            "detailed_results_df_by_instruction_dict": detailed_results_df_by_instruction_dict,
            "wrong_questions_from_start_counter": wrong_questions_from_start_counter,
        }    
        
        # prepare data-related keyword arguments
        data_kwargs = {
            "raw_data": raw_data,
            "train_index": train_index,
            "eval_index": eval_index,
            "prediction_treat_as_number": prediction_treat_as_number,
            "prediction_treat_as_bool": prediction_treat_as_bool,
            "prediction_treat_as_text": prediction_treat_as_text,
            "is_multiple_choice": is_multiple_choice,
            "is_multiple_choice_eval": is_multiple_choice_eval,
        }
        return data_kwargs, result_kwargs
        
    def _evaluate_and_analyze(
            self,
            *,
            raw_answers: Sequence[str],
            true_answers: Sequence[str],
            raw_prompts_flattened: Sequence[str],
            instruction: str,
            prediction_treat_as_number: bool | Literal["adaptive"],
            prediction_treat_as_bool: bool,
            prediction_treat_as_text: bool,
            is_multiple_choice_all: bool | List[bool],
            ) -> Tuple[Sequence, Sequence]:
        
        logger.info(f"""computing the score of "{instruction}" by prompting""")
        num_prediction = len(raw_answers)
        if isinstance(is_multiple_choice_all, bool):
            is_multiple_choice_all = [is_multiple_choice_all] * num_prediction
        else:
            assert (
                    len(is_multiple_choice_all) == num_prediction
            ), "is_multiple_choice must have the same length as raw_answers"
        if self.verbose:
            logger.info(
                "extracting final prediction with"
                f" treat_as_number={prediction_treat_as_number},"
                f" treat_as_bool={prediction_treat_as_bool},"
                f" treat_as_text={prediction_treat_as_text}, and"
                f" num_decimals={self.prediction_num_decimals}"
            )

        # Based on specific formats of the second-round answers, the function below
        # extracts the corresponding texts for parsing. Here're roles of all parts:
        # .strip(":") - following "the answer is", some answers have ":" at the
        # beginning
        # .strip() - some answers have "\n" or blank spaces at the beginning, or have
        # "\n" after ":"
        # .split("\n")[0] - extract the texts before the first "\n\n" after the above
        # stripping
        # .split("Q:")[0] - extract the texts before "Q:" after the above stripping
        def _extract_second_round_answer_for_parsing(ans):
            return ans.strip(":").strip().split("\n")[0].split("Q:")[0]

        raw_answers_to_parse = (
            list(  # pylint: disable=g-long-ternary
                map(
                    _extract_second_round_answer_for_parsing, raw_answers
                )
            )
            if self.extract_final_answer_by_prompting_again
            else raw_answers
        )

        if prediction_treat_as_number == "adaptive":
            true_answer_is_numeric = [item.isnumeric() for item in true_answers]
            prediction_treat_as_number_list = true_answer_is_numeric.copy()
        else:
            assert isinstance(prediction_treat_as_number, bool)
            prediction_treat_as_number_list = [prediction_treat_as_number] * len(
                true_answers
            )

        def _parse_prediction(
                x, is_gpt_model, treat_as_number, num_decimals, treat_as_bool, treat_as_text=False
        ):
            if treat_as_text:
                return x
            # import pdb; pdb.set_trace()
            if is_gpt_model and r"\boxed" in x:
                return re.findall(r"\\boxed{(.*?)}", x)[0]
            else:
                return metrics.get_normalized_prediction(
                    x,
                    treat_as_number=treat_as_number,
                    num_decimals=num_decimals,
                    treat_as_bool=treat_as_bool,
                )

        def _parse_prediction_with_star(prediction):
            try:
                return re.findall(r"\*\*(.*?)\*\*", prediction)[-1]
            except Exception:
                return "-1"

        # pylint: disable=g-long-lambda
        text_in_star = list(
            map(
                lambda x: _parse_prediction_with_star(
                    x,
                ),
                raw_answers_to_parse,
            )
        )

        # pylint: disable=g-long-lambda
        # import pdb; pdb.set_trace()
        choices = list(
            map(
                lambda x, y: _parse_prediction(
                    x,
                    isinstance(self.scorer_llm, OpenaiLLM),
                    y,
                    self.prediction_num_decimals,
                    prediction_treat_as_bool,
                    prediction_treat_as_text,
                ),
                text_in_star,
                prediction_treat_as_number_list,
            )
        )

        if not self.extract_final_answer_by_prompting_again:
            choices = [
                _extract_second_round_answer_for_parsing(item) for item in choices
            ]
                  
        accuracies = []
        # def _evaluate_score(p, r, i, t):
        #     return self.evaluator.evaluate_strings(
        #         prediction=p,
        #         reference=r,
        #         input=i,
        #         treat_include_as_correct=t
        #     )
        # import pdb; pdb.set_trace()
        for i, _ in enumerate(raw_answers):
            treat_include_as_correct = not prediction_treat_as_number_list[i]
            input_text = raw_prompts_flattened[i] if is_multiple_choice_all[i] else ""
            scores = self.evaluator.evaluate_strings(
                prediction=choices[i],
                reference=true_answers[i],
                input=input_text,
                treat_include_as_correct=treat_include_as_correct
            )
            accuracies.append(scores['score'])
            
        return choices, accuracies

    def metadata_to_df(
            self,
            *,
            eval_index_all: Sequence,
            raw_prompts_flattened: Sequence,
            raw_answers: Sequence,
            raw_answers_second_round: Sequence | None = None,
            raw_prompts_flattened_second_round: Sequence | None = None,
            choices: Sequence,
            true_answers: Sequence,
            accuracies: Sequence,
            ) -> pd.DataFrame:
        detailed_results_df = pd.DataFrame(
            list(
                zip(
                    eval_index_all,
                    raw_prompts_flattened,
                    raw_answers,
                    choices,
                    true_answers,
                    accuracies,
                )
            ),
            columns=[
                "index_in_raw_dataset",
                "raw_prompt",
                "raw_answer",
                "parsed_answer",
                "true_answer",
                "accuracy",
            ],
        )
        if self.extract_final_answer_by_prompting_again:
            detailed_results_df.insert(
                3, "raw_prompt_second_round", raw_prompts_flattened_second_round
            )
            detailed_results_df.insert(
                4, "raw_answer_second_round", raw_answers_second_round
            )

        detailed_results_df.set_index("index_in_raw_dataset", inplace=True)
        return detailed_results_df
    
    def run(self) -> Dict[str, Any]:
        """
		The primary method initiating and orchestrating the evolutionary process for optimizing instructions.
		It configures settings, prints experimental parameters, initializes dataset specifics, and starts
		with the evaluation of initial instructions using a scoring language model.

		This includes handling different dataset types (e.g., multiple-choice, open-ended),
		dynamically adjusting configurations based on dataset requirements, and setting up the environment
		for iterative instruction improvement steps.
		"""
        data_kwargs, result_kwargs = self._before_run()
        for i_step in range(self.num_steps):
            logger.info(f"\n================== Step {i_step} =====================")
            if not i_step % 10:
                old_instructions_and_scores = result_kwargs["old_instructions_and_scores"]
                logger.info(f"old_instructions_and_scores: {old_instructions_and_scores}")
            if self.optimizer_llm_temperature_schedule == "linear_increase":
                optimizer_llm_temperature_curr = (
                    self.optimizer_llm_temperature
                    + i_step
                    / self.num_steps
                    * (self.optimizer_llm_temperature_end - self.optimizer_llm_temperature)
                )
            else:
                optimizer_llm_temperature_curr = self.optimizer_llm_temperature
            result_kwargs, early_stop = self._step(
                i_step=i_step, 
                optimizer_llm_temperature=optimizer_llm_temperature_curr, 
                **data_kwargs, 
                **result_kwargs)
            if early_stop:
                break
        return self._after_run()
    
    def _after_run(self):
        best_prompt = self.extract_best_prompt(self.store_path)
        return best_prompt
    def _predict(
            self,
            eval_index_all: Sequence,
            raw_data: Sequence,
            instruction: str,
            ) -> Dict[str, Sequence]:
        # generate raw prompts
        raw_prompts_flattened = []
        raw_prompts_flattened_second_round, raw_answers_second_round = [], []
        num_eval_examples = len(eval_index_all)
        for i in range(num_eval_examples):
            raw_prompt = gen_prompt(
                raw_data,
                instruction=instruction,
                idx=eval_index_all[i],
                include_qa=self.include_qa,
                instruction_pos=self.instruction_pos,
                dataset_name=self.dataset_name,
                task_name=self.task_name
            )
            raw_prompts_flattened.append(raw_prompt)
            # import pdb;pdb.set_trace()
        if self.evaluate_in_parallel:
            raw_answers = asyncio.run(
                async_call_llm(llm=self.scorer_llm, list_of_messages=raw_prompts_flattened, semaphore=self.semaphore)
                )
        else:
            from tqdm import tqdm  # no parallelism in first round
            raw_answers = [
                self.scorer_llm.chat(messages=prompt).message.content
                for prompt in tqdm(raw_prompts_flattened)
            ]
            # for prompt in tqdm(raw_prompts_flattened):
            #     print(self.scorer_llm.chat(messages=prompt).message.content)


        if self.verbose:
            logger.info("first round of prompting finished")

        # prompt again to better extract answers
        if self.extract_final_answer_by_prompting_again:
            raw_prompts_flattened_second_round = list(
                map(
                    lambda a, b: a + " " + _split_by_Q(b),
                    raw_prompts_flattened,
                    raw_answers,
                )
            )
            raw_prompts_flattened_second_round = [
                item + " " + "So the final answer is"
                for item in raw_prompts_flattened_second_round
            ]

            # second round of prompting to extract final answer
            # We only need a small max_decode_steps because the answer usually shows up
            # at the very beginning of the output. The decode length can't be too small
            # though, because on some GSM8K questions the second-round answers include
            # some calculations before arriving at the final answer
            if self.evaluate_in_parallel:
                raw_answers_second_round = asyncio.run(
                    async_call_llm(llm=self.scorer_llm, list_of_messages=raw_prompts_flattened_second_round, semaphore=self.semaphore)
                    )
            else:
                from tqdm import tqdm  # no parallelism in first round
                raw_answers_second_round = [
                    self.scorer_llm.chat(messages=prompt).message.content
                    for prompt in tqdm(raw_prompts_flattened_second_round)
                ]
        # import pdb;pdb.set_trace()
        if self.verbose:
            print("second round of prompting finished")
        return {"raw_answers": raw_answers,
                "raw_answers_second_round": raw_answers_second_round,
                "raw_prompts_flattened": raw_prompts_flattened,
                "raw_prompts_flattened_second_round": raw_prompts_flattened_second_round,
                }
    
    def _update_prompt(self,
                       *,
                       i_step: int,
                       meta_prompt: str,
                       temperature: float,
                       old_instruction_md5_hashstrings_set: Set,
                       ) -> Sequence[str]:
        generated_instructions_raw = []

        optimizer_llm_input_texts = [meta_prompt] * self.num_generated_instructions_in_each_step
        if not self.generate_in_parallel:
            from tqdm import tqdm
            raw_outputs = [self.optim_llm.chat(
                messages=optimizer_llm_input_text, temperature=temperature).message.content
                for optimizer_llm_input_text in tqdm(optimizer_llm_input_texts)]
                # print(raw_outputs)
        else:
            raw_outputs = asyncio.run(
                async_call_llm(llm=self.scorer_llm, 
                               list_of_messages=optimizer_llm_input_texts, 
                               temperature=temperature,
                               semaphore=self.semaphore)
                )
        # logger.info(raw_outputs)
        # import pdb; pdb.set_trace()
        # Extract the generated instructions from the optimizer LLM output. Only
        # keep some samples if the desired number of remaining instructions
        # is smaller than the total number of decodes in this step.
        if self.meta_prompt_type == "both_instructions_and_exemplars":
            if self.optim_llm_name.lower() in [e.value for e in OpenaiLlmModel]:
                if self.instruction_pos == "A_begin":
                    start_string = "<Start>"
                    end_string = "</Start>"
                else:
                    start_string = "<INS>"
                    end_string = "</INS>"
                for raw_output in raw_outputs:
                    if start_string not in raw_output:
                        start_index = 0
                    else:
                        start_index = raw_output.index(start_string) + len(start_string)
                    if end_string not in raw_output:
                        end_index = len(raw_output)
                    else:
                        end_index = raw_output.index(end_string)
                    new_inst = raw_output[start_index:end_index].strip()
                    generated_instructions_raw.append(new_inst)
            else:
                generated_instructions_raw += [
                    extract_string_in_square_brackets(string)
                    for string in raw_outputs
                ]
        else:
            assert self.meta_prompt_type == "instructions_only"
            max_num_instructions_to_keep_in_each_output = 1
            for string in raw_outputs:
                generated_instructions_raw += parse_tag_content(string)[
                                              :max_num_instructions_to_keep_in_each_output
                                              ]
        # import pdb; pdb.set_trace()
        generated_instructions_raw = list(
            map(polish_sentence, generated_instructions_raw)
        )
        logger.info(f"\ninitially generated instructions: {generated_instructions_raw}\n")
        # do not evaluate old instructions again
        generated_instructions = []  # the new instructions generated in this step
        for ins in generated_instructions_raw:
            ins_md5_hashstring = instruction_to_filename(
                ins, md5_hashing=True
            )
            if ins_md5_hashstring not in old_instruction_md5_hashstrings_set:
                generated_instructions.append(ins)
                old_instruction_md5_hashstrings_set.add(ins_md5_hashstring)
            else:
                print(f"already evaluated '{ins}' previously")
        generated_instructions = list(set(generated_instructions))

        to_evaluate_instructions = []
        for instruction in generated_instructions:
            if len(instruction) > 500:
                print(f"Step {i_step}, instruction: {instruction}, too long, skipped")
                continue
            if not instruction:
                print(f"Step {i_step}, instruction: {instruction}, empty, skipped")
                continue
            if self.dataset_name == "gsm8k" and any(
                    char.isdigit() for char in instruction
            ):
                print(
                    f"Step {i_step}, instruction: {instruction}, contains numbers,"
                    " skipped"
                )
                continue
            if "INS" in instruction:
                print(
                    f"Step {i_step}, instruction: {instruction}, contains 'INS',"
                    " skipped"
                )
                continue
            to_evaluate_instructions.append(instruction)
        logger.info(f"\nto-evaluate generated instructions: {to_evaluate_instructions}\n")
        return old_instruction_md5_hashstrings_set, to_evaluate_instructions, generated_instructions_raw

    def _step(self,
              *,
             i_step,
             optimizer_llm_temperature: float,
             ## result_kwargs
             meta_prompts,
             old_instructions_and_scores: Sequence,
             old_instructions_and_scores_raw: Sequence,
             wrong_questions_from_start_counter: collections.Counter,
             few_shot_index_list_by_step_dict: Dict[int, Any],
             old_instruction_md5_hashstrings_set: Set,
             prev_saved_instructions: Sequence,
             generated_ins_on_few_shot_results_dict: Dict[int, Any],
             old_ins_on_few_shot_results_dict: Dict[int, Any],
             detailed_results_df_by_instruction_dict: Dict[str, Any],
             eval_detailed_results_df_by_instruction_dict: Dict[str, Any],
             instruction_score_dict: Dict[str, Any],
             instruction_eval_score_dict: Dict[str, Any],
             eval_results: Sequence,
             ## data_kwargs
             raw_data,
             train_index: Sequence,
             eval_index: Sequence,
             prediction_treat_as_number: bool | Literal["adaptive"],
             prediction_treat_as_bool: bool,
             prediction_treat_as_text: bool,
             is_multiple_choice: bool | List[bool],
             is_multiple_choice_eval: bool | List[bool],
             ) -> Tuple[Dict[str, Any], bool]:
        """
		Generates new instructions based on different selection criteria for few-shot examples.

		This function dynamically adjusts the few-shot examples presented to the model during the
		instruction refinement process. It supports various strategies for selecting these examples,
		intended to improve the model's understanding and performance on tasks by focusing on areas
		of weakness or through random exposure for diversity.

		Conditions covered:
		- 'accumulative_most_frequent': Selects questions that have been answered incorrectly most frequently
		  since the start, either all or a sampled subset depending on their total count.
		- 'current_most_frequent': Chooses questions that are frequently missed under the current set of
		  instructions, ensuring diversity if not enough unique questions meet the threshold.
		- 'constant': Picks a constant set of few-shot questions randomly but fixed across iterations.
		- 'random': Randomly selects questions for each iteration, introducing variability.

		Args:
			few_shot_qa_pairs (bool): Flag indicating whether to use few-shot QA pairs in instruction generation.
			few_shot_selection_criteria (str): The strategy for picking few-shot examples ('accumulative_most_frequent',
										  'current_most_frequent', 'constant', 'random').
			Other variables used (not passed as arguments):
				self.wrong_questions_from_start_counter: A counter tracking the frequency of incorrect answers.
				num_few_shot_questions_for_instruction_refinement: The number of few-shot examples to include.
				i_step: The current iteration step, relevant for seeded randomness.
				old_instructions_and_scores: Historical data on instruction performance.
				old_instruction_score_threshold: A threshold for considering instruction performance.
				result_by_instruction_folder: Directory containing results per instruction.
				max_num_instructions, num_score_buckets: Parameters for instruction scoring.
				self.train_index: List of indices representing the training dataset.

		Returns (implicitly through assignment):
			few_shot_index_list (List[int]): Indices of questions selected for the few-shot demonstration.
		"""
        # generate new instructions
        if self.few_shot_qa_pairs:
            if self.few_shot_selection_criteria == "accumulative_most_frequent":
                # select QA pairs that were done wrong the most number of times
                most_frequent_wrong_question_indices = [
                    k
                    for k, _ in sorted(
                        wrong_questions_from_start_counter.items(), key=lambda x: -x[1]
                    )
                ]
                print(
                    "len(most_frequent_wrong_question_indices):"
                    f" {len(most_frequent_wrong_question_indices)}"
                )
                if (
                        len(most_frequent_wrong_question_indices)
                        <= self.num_few_shot_questions_for_instruction_refinement
                ):
                    few_shot_index_list = most_frequent_wrong_question_indices.copy()
                else:
                    np.random.seed(i_step)
                    few_shot_index_list = np.sort(
                        np.random.choice(
                            most_frequent_wrong_question_indices,
                            self.num_few_shot_questions_for_instruction_refinement,
                            replace=False,
                        )
                    )

            elif self.few_shot_selection_criteria == "current_most_frequent":
                # show exemplars done wrong most often by currently shown instructions
                old_instruction_score_threshold_single_step = (
                    self.old_instruction_score_threshold if i_step > 0 else 0
                )
                _, old_instructions_and_scores_in_meta_prompt = (
                    gen_ins_and_score_pairs_substr(
                        old_instructions_and_scores=old_instructions_and_scores,
                        old_instruction_score_threshold=old_instruction_score_threshold_single_step,
                        max_num_instructions=self.max_num_instructions,
                        return_str_only=False,
                        num_score_buckets=self.num_score_buckets,
                    )
                )
                wrong_questions_counter_single_step = collections.Counter()
                for ins, _, _ in old_instructions_and_scores_in_meta_prompt:
                    filename = instruction_to_filename(ins)
                    file_path = os.path.join(
                        self.store_path, f"{filename}.csv"
                    )
                    single_ins_df = pd.read_csv(file_path, index_col=0, header=0)
                    wrong_question_indices_set_single_old_ins = set(
                        list(
                            single_ins_df.iloc[
                            np.where(single_ins_df.accuracy == 0.0)[0], :
                            ].index
                        )
                    )
                    for idx in wrong_question_indices_set_single_old_ins:
                        wrong_questions_counter_single_step[idx] += 1
                most_occurred_wrong_questions = [
                    k
                    for k, v in wrong_questions_counter_single_step.items()
                    if v == max(wrong_questions_counter_single_step.values())
                ]
                if (
                        len(most_occurred_wrong_questions)
                        < self.num_few_shot_questions_for_instruction_refinement
                ):
                    # pylint: disable=cell-var-from-loop
                    idx_most_to_least = sorted(
                        wrong_questions_counter_single_step,
                        key=lambda x: -wrong_questions_counter_single_step[x],
                    )
                    few_shot_index_list = idx_most_to_least[
                                          :self.num_few_shot_questions_for_instruction_refinement
                                          ]
                else:
                    few_shot_index_list = np.sort(
                        np.random.choice(
                            most_occurred_wrong_questions,
                            self.num_few_shot_questions_for_instruction_refinement,
                            replace=False,
                        )
                    )
            elif self.few_shot_selection_criteria == "constant":
                np.random.seed(0)
                few_shot_index_list = np.sort(
                    np.random.choice(
                        train_index,
                        self.num_few_shot_questions_for_instruction_refinement,
                        replace=False,
                    )
                )
            else:
                assert self.few_shot_selection_criteria == "random"
                np.random.seed(i_step)
                few_shot_index_list = np.sort(
                    np.random.choice(
                        train_index,
                        self.num_few_shot_questions_for_instruction_refinement,
                        replace=False,
                    )
                ).tolist()
            few_shot_index_list_by_step_dict[i_step] = few_shot_index_list
            meta_prompt = gen_meta_prompt(
                old_instructions_and_scores=old_instructions_and_scores,
                instruction_pos=self.instruction_pos,
                optimizer_llm_name=self.optim_llm_name,
                old_instruction_score_threshold=self.old_instruction_score_threshold,
                max_num_instructions=self.max_num_instructions,
                meta_prompt_type=self.meta_prompt_type,
                few_shot_qa_pairs=True,
                include_qa=self.include_qa,
                data=raw_data,
                few_shot_index_list=few_shot_index_list,
                instructions_before_exemplars=self.meta_prompt_instructions_before_exemplars,
                num_score_buckets=self.num_score_buckets,
                dataset_name=self.dataset_name,
                task_name=self.task_name,
                prompt_handler=self.prompt_handler,
            )

        else:  # no few-shot exemplars in meta-prompt
            few_shot_index_list = []
            meta_prompt = gen_meta_prompt(
                old_instructions_and_scores=old_instructions_and_scores,
                instruction_pos=self.instruction_pos,
                optimizer_llm_name=self.optimizer_llm_name,
                old_instruction_score_threshold=self.old_instruction_score_threshold,
                max_num_instructions=self.max_num_instructions,
                meta_prompt_type=self.meta_prompt_type,
                few_shot_qa_pairs=False,
                include_qa=self.include_qa,
                instructions_before_exemplars=self.meta_prompt_instructions_before_exemplars,
                num_score_buckets=self.num_score_buckets,
                dataset_name=self.dataset_name,
                task_name=self.task_name,
                prompt_handler=self.prompt_handler
            )
        logger.info(f"\nmeta_prompt: \n\n{meta_prompt}\n")
        meta_prompts.append((meta_prompt, i_step))

        old_instruction_md5_hashstrings_set, to_evaluate_instructions, generated_instructions_raw = \
        self._update_prompt(
            i_step=i_step, meta_prompt=meta_prompt, temperature=optimizer_llm_temperature, 
            old_instruction_md5_hashstrings_set=old_instruction_md5_hashstrings_set
            )
        
        # evaluate new instructions on the few-shot exemplars in meta-prompt
        if self.few_shot_qa_pairs and self.evaluate_generated_ins_on_few_shot:
            print("evaluating GENERATED instructions on few-shot exemplars")
            single_step_eval_on_few_shot = dict()
            true_answers = [
                str(fetch_true_answer(raw_data, idx=idx, dataset_name=self.dataset_name))
                for idx in few_shot_index_list]
            for instruction in to_evaluate_instructions:
                if instruction not in prev_saved_instructions:
                    print(
                        f"evaluating Step {i_step}, instruction: {instruction} on"
                        " few-shot exemplars"
                    )
                detailed_results_df = self.evaluate_single_instruction(
                    instruction=instruction, 
                    raw_data=raw_data,
                    index_to_evaluate=few_shot_index_list,
                    true_answers=true_answers,
                    prediction_treat_as_number=prediction_treat_as_number,
                    prediction_treat_as_bool=prediction_treat_as_bool,
                    prediction_treat_as_text=prediction_treat_as_text,
                    is_multiple_choice=is_multiple_choice,
                )
                single_step_eval_on_few_shot[instruction] = detailed_results_df
                
            logger.info(
                f"Step {i_step}, single_step_eval_on_few_shot:"
                f" {single_step_eval_on_few_shot}\n"
            )
            generated_ins_on_few_shot_results_dict[i_step] = (
                single_step_eval_on_few_shot
            )

        # evaluate OLD instructions on the few-shot exemplars in meta-prompt
        if self.few_shot_qa_pairs and self.evaluate_old_ins_on_few_shot:
            logger.info("evaluating OLD instructions on few-shot exemplars")
            single_step_eval_on_few_shot = dict()
            true_answers = [
                str(fetch_true_answer(raw_data, idx=idx, dataset_name=self.dataset_name))
                for idx in few_shot_index_list]
            for instruction, _, _ in old_instructions_and_scores:
                logger.info(
                    f"evaluating Step {i_step}, instruction: {instruction} on few-shot"
                    " exemplars"
                )
                detailed_results_df = self.evaluate_single_instruction(
                    instruction=instruction, 
                    raw_data=raw_data,
                    index_to_evaluate=few_shot_index_list,
                    true_answers=true_answers,
                    prediction_treat_as_number=prediction_treat_as_number,
                    prediction_treat_as_bool=prediction_treat_as_bool,
                    prediction_treat_as_text=prediction_treat_as_text,
                    is_multiple_choice=is_multiple_choice,
                )
                single_step_eval_on_few_shot[instruction] = detailed_results_df

            logger.info(
                f"Step {i_step}, single_step_eval_on_few_shot:"
                f" {single_step_eval_on_few_shot}\n"
            )
            old_ins_on_few_shot_results_dict[i_step] = single_step_eval_on_few_shot

        # evaluate newly generated instructions on the training set
        true_answers = [
            str(fetch_true_answer(raw_data, idx=idx, dataset_name=self.dataset_name))
            for idx in train_index]
        for instruction in to_evaluate_instructions:
            if instruction not in prev_saved_instructions:
                print(f"""computing the score of "{instruction}" by prompting""")
                detailed_results_df = self.evaluate_single_instruction(
                    instruction=instruction, 
                    raw_data=raw_data,
                    index_to_evaluate=train_index,
                    true_answers=true_answers,
                    prediction_treat_as_number=prediction_treat_as_number,
                    prediction_treat_as_bool=prediction_treat_as_bool,
                    prediction_treat_as_text=prediction_treat_as_text,
                    is_multiple_choice=is_multiple_choice,
                )
                prev_saved_instructions.add(instruction)
            else:
                # do not re-evaluate instructions that had been evaluated previously
                detailed_results_df = pd.read_csv(
                    os.path.join(self.store_path, f"{instruction}.csv"),
                    index_col=0,
                    header=0,
                )
                logger.info(f"""reading previously saved "{instruction}" information""")

            scores = detailed_results_df["accuracy"]
            average_score = np.average(scores)
            logger.info(
                f"Step {i_step}, instruction: {instruction}, score: {average_score}"
            )

            # increment the counter on wrong questions
            wrong_question_indices_set = set(
                list(
                    detailed_results_df[detailed_results_df["accuracy"] == 0.0].index
                )
            )
            for idx in wrong_question_indices_set:
                wrong_questions_from_start_counter[idx] += 1

            filename = instruction_to_filename(instruction)
            file_path = os.path.join(
                self.store_path, f"""{filename}.csv"""
            )
            detailed_results_df.to_csv(file_path, index=True, header=True)
            logger.info(f"saving results to {file_path}")

            detailed_results_df_by_instruction_dict[instruction] = detailed_results_df
            old_instructions_and_scores.append((instruction, average_score, i_step))
            instruction_score_dict[instruction] = average_score

        # record all generated instructions
        for instruction in generated_instructions_raw:
            if instruction in instruction_score_dict:
                average_score = instruction_score_dict[instruction]
            else:
                average_score = np.nan
                old_instructions_and_scores_raw.append(
                    (instruction, average_score, i_step)
                )

        # =============================== eval ====================================
        # every eval_interval steps, evaluate the instructions that were generated
        # in the current step and were not skipped
        if not i_step % self.eval_interval and len(eval_index) > 0:
            true_answers = [
                str(fetch_true_answer(raw_data, idx=idx, dataset_name=self.dataset_name))
                for idx in eval_index]
            for instruction in generated_instructions_raw:
                # if the instruction wasn't skipped in any step
                if instruction in instruction_score_dict:
                    if instruction not in instruction_eval_score_dict:
                        detailed_results_df = self.evaluate_single_instruction(
                            instruction=instruction, 
                            raw_data=raw_data,
                            index_to_evaluate=eval_index,
                            true_answers=true_answers,
                            prediction_treat_as_number=prediction_treat_as_number,
                            prediction_treat_as_bool=prediction_treat_as_bool,
                            prediction_treat_as_text=prediction_treat_as_text,
                            is_multiple_choice=is_multiple_choice_eval,
                        )
                        eval_score = np.average(detailed_results_df["accuracy"])
                        eval_detailed_results_df_by_instruction_dict[instruction] = detailed_results_df
                        instruction_eval_score_dict[instruction] = eval_score
                    else:
                        eval_score = instruction_eval_score_dict[instruction]
                    logger.info(
                        f"EVAL: \nStep {i_step}, instruction: {instruction}, eval score:"
                        f" {eval_score:.2f}"
                    )
                    eval_results.append((i_step, instruction, eval_score))

        # ===================== save up-to-date results ===========================
        results_dict = dict()
        results_dict["meta_prompts"] = meta_prompts
        results_dict["old_instructions_and_scores"] = list(
            old_instructions_and_scores
        )
        results_dict["old_instructions_and_scores_raw"] = list(
            old_instructions_and_scores_raw
        )
        results_dict["generated_ins_on_few_shot_results_dict"] = (
            generated_ins_on_few_shot_results_dict
        )
        results_dict["old_ins_on_few_shot_results_dict"] = (
            old_ins_on_few_shot_results_dict
        )
        results_dict["few_shot_index_list_by_step_dict"] = (
            few_shot_index_list_by_step_dict
        )
        results_dict["eval_results"] = eval_results
        results_dict["eval_detailed_results_df_by_instruction_dict"] = (
            eval_detailed_results_df_by_instruction_dict
        )
        with open(os.path.join(self.store_path, "results_dict.pkl"), "wb") as fp:
            pkl.dump(results_dict, fp)
        logger.info(f"\nsaved all results to\n{self.store_path}")
        
        results_dict["wrong_questions_from_start_counter"] = (
            wrong_questions_from_start_counter
        )
        results_dict["old_instruction_md5_hashstrings_set"] = (
            old_instruction_md5_hashstrings_set
        )
        results_dict["prev_saved_instructions"] = (
            prev_saved_instructions
        )
        results_dict["detailed_results_df_by_instruction_dict"] = (
            detailed_results_df_by_instruction_dict
        )
        results_dict["instruction_score_dict"] = (
            instruction_score_dict
        )
        results_dict["instruction_eval_score_dict"] = (
            instruction_eval_score_dict
        )
        results_dict["eval_results"] = (
            eval_results
        )
        # TODO by yunze: early stop
        return results_dict, False

    def extract_best_prompt(self, save_folder) -> str:
        """
		This method is intended to extract the best prompt from the saved evaluated results.
		"""
        with open(os.path.join(save_folder, "results_dict.pkl"), "rb") as fp:
            results_dict = pkl.load(fp)
        return sorted(results_dict['old_instructions_and_scores'])[-1][0]
