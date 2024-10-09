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

import numpy as np
import pandas as pd

from meta_icl import CONFIG_REGISTRY
from meta_icl.algorithm.base_algorithm import PromptOptimizationWithFeedback
from meta_icl.algorithm.opro.evaluation import eval_utils
from meta_icl.algorithm.opro.optimization import opt_utils
from meta_icl.core.models.generation_model import (
    AioGenerationModel,
    GenerationModel,
    OpenAIAioGenerationModel,
    OpenAIAioPostModel,
    OpenAIGenerationModel,
    OpenAIPostModel
)


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
    FILE_PATH: str = __file__

    def __init__(self, language="cn", **kwargs):
        """
		Initializes the OPRO instance with necessary configurations, models, and data structures for
		tracking optimization progress.

		Args:
			language (str): The target language for prompt optimization.
			**kwargs: Additional keyword arguments for configuration overrides.
		"""
        super().__init__(language=language, **kwargs)
        self.init_config()
        self.init_model()
        self.scorer_llm_name = self.model_config.scorer.model_name
        self.optimizer_llm_name = self.model_config.optim.model_name
        self.dataset_name = self.basic_config.dataset_name.lower()
        self.task_name = self.basic_config.task_name
        self.meta_prompt_type = self.basic_config.meta_prompt_type
        self.instruction_pos = self.basic_config.instruction_pos
        # the dictionary of the few-shot QA indices in meta-prompt
        # key: step index; value: the list of few-shot indices in that step
        self.few_shot_index_list_by_step_dict = dict()
        self.generated_ins_on_few_shot_results_dict = dict()
        self.old_ins_on_few_shot_results_dict = dict()
        # evaluation results every a few steps
        # format: [(i_step, instruction, detailed_results_df)]
        self.eval_results = []
        # all generated instructions, format: [(instruction, score, step_index)]
        # the instructions that were skipped have score NaN
        self.old_instructions_and_scores_raw = []
        # the new instructions, format: [(instruction, score, step_index)]
        self.old_instructions_and_scores = []
        self.meta_prompts = []  # format: [(meta_prompt, step_index)]
        self.instruction_score_dict = dict()  # the dictionary of {instruction: score}

        self.detailed_results_df_by_instruction_dict = dict()
        self.wrong_questions_from_start_counter = collections.Counter()
        # EVAL results
        self.eval_detailed_results_df_dict = dict()  # {instruction: detailed_results_df}
        self.instruction_eval_score_dict = dict()  # {instruction: eval_score}
        self.old_instruction_md5_hashstrings_set = set()
        self.prev_saved_instructions = set()

    def init_model(self):
        """
		Initializes the language models used for scoring and optimization based on the configuration.
		If parallel evaluation is enabled, asynchronous models are initialized; otherwise, synchronous models are used.
		"""
        scorer_module_name = self.model_config.scorer.get('module_name')
        optim_module_name = self.model_config.optim.get('module_name')

        if self.evolution_config.evaluate_in_parallel:
            if scorer_module_name == 'aio_generation':
                self.scorer_llm = AioGenerationModel(**self.model_config.scorer)
            elif scorer_module_name == 'openai_aio_generation':
                self.scorer_llm = OpenAIAioGenerationModel(**self.model_config.scorer)
            elif scorer_module_name == 'openai_aio_post':
                self.scorer_llm = OpenAIAioPostModel(**self.model_config.scorer)

            if optim_module_name == 'aio_generation':
                self.optim_llm = AioGenerationModel(**self.model_config.optim)
            elif optim_module_name == 'openai_aio_generation':
                self.optim_llm = OpenAIAioGenerationModel(**self.model_config.optim)
            elif optim_module_name == 'openai_aio_post':
                self.optim_llm = OpenAIAioPostModel(**self.model_config.optim)
        else:
            if scorer_module_name == 'dashscope_generation':
                self.scorer_llm = GenerationModel(**self.model_config.scorer)
            elif scorer_module_name == 'openai_generation':
                self.scorer_llm = OpenAIGenerationModel(**self.model_config.scorer)
            elif scorer_module_name == 'openai_post':
                self.scorer_llm = OpenAIPostModel(**self.model_config.scorer)

            if optim_module_name == 'dashscope_generation':
                self.optim_llm = GenerationModel(**self.model_config.optim)
            elif optim_module_name == 'openai_generation':
                self.optim_llm = OpenAIGenerationModel(**self.model_config.optim)
            elif optim_module_name == 'openai_post':
                self.optim_llm = OpenAIPostModel(**self.model_config.optim)

    def init_config(self):
        """
		Initialize configuration settings for the OPRO system by retrieving
		configurations from the registry for the model, task, basic settings,
		and evolution strategies. This method sets up the essential parameters
		for the entire optimization process.

		The configurations include:
		- Model configuration: Details about the language model used (e.g., Qwen models).
		- Task configuration: Settings specific to the tasks like MMLU, BBH, GSM8K.
		- Basic configuration: Fundamental operational settings of the system.
		- Evolution configuration: Strategies for iterative improvement of prompts.

		Returns:
			None. Configurations are stored as attributes within the instance.
		"""
        self.model_config = CONFIG_REGISTRY.module_dict['model_config']
        self.basic_config = CONFIG_REGISTRY.module_dict['basic_config']
        self.evolution_config = CONFIG_REGISTRY.module_dict["evolution_config"]

    def update_config(self, config='evolution_config', **kwargs):
        """
		Updates the specified configuration dictionary with the provided key-value pairs.

		Args:
			config (str): The name of the configuration attribute to update. Defaults to 'evolution_config'.
			**kwargs: Arbitrary keyword arguments representing the key-value pairs to update in the config.

		Note:
			This method uses `__getattribute__` to dynamically access the attribute corresponding to the given `config`.
		"""
        self.__getattribute__(config).update(**kwargs)

    def run(self):
        """
		The primary method initiating and orchestrating the evolutionary process for optimizing instructions.
		It configures settings, prints experimental parameters, initializes dataset specifics, and starts
		with the evaluation of initial instructions using a scoring language model.

		This includes handling different dataset types (e.g., multiple-choice, open-ended),
		dynamically adjusting configurations based on dataset requirements, and setting up the environment
		for iterative instruction improvement steps.
		"""
        print(f"tasks_all: {self.evolution_config.tasks_all}")
        print(
            f"train_ratio: {self.evolution_config.train_ratio}, number of training points:"
            f" {int(self.evolution_config.num_examples * self.evolution_config.train_ratio)}"
        )
        print(
            f"eval_ratio: {self.evolution_config.eval_ratio}, number of eval points: "
            f"{int(self.evolution_config.num_examples * self.evolution_config.eval_ratio)}"
        )
        print(
            f"test_ratio: {self.evolution_config.test_ratio}, number of test points: "
            f"{int(self.evolution_config.num_examples * self.evolution_config.test_ratio)}"
        )
        print(
            f"generating {self.evolution_config.num_generated_instructions_in_each_step} instructions in"
            f" each step, run for {self.evolution_config.num_search_steps} steps"
        )
        print(
            "discarding generated instructions with score less than:"
            f" {self.evolution_config.old_instruction_score_threshold} (old_instruction_score_threshold)"
        )
        print(f"num_score_buckets: {self.evolution_config.num_score_buckets}")

        # ================= experiment configurations =============================
        if "optimizer_llm_temperature_schedule" not in self.evolution_config:
            self.evolution_config["optimizer_llm_temperature_schedule"] = "constant"
        assert self.evolution_config["optimizer_llm_temperature_schedule"] in {
            "constant",
            "linear_increase",
        }, "The temperature schedule should be constant or linear_increase."

        if "optimizer_llm_temperature_end" not in self.evolution_config:
            self.evolution_config["optimizer_llm_temperature_schedule"] = None
        if "verbose" not in self.evolution_config:
            self.evolution_config["verbose"] = False

        # =================== save configurations to json file ====================
        # import json
        # with open(os.path.join(self.evolution_config.save_folder, "configs_dict.json"), "w") as f:
        # 	json.dump(dict(self.evolution_config), f, indent=4)
        self.train_index = self.evolution_config.train_index
        self.eval_index = self.evolution_config.eval_index
        # todo: by jm, how to optimize the prompt for other dataset or general applications.
        if self.dataset_name == "mmlu":
            self.is_multiple_choice = True
            self.is_multiple_choice_eval = True
        elif self.dataset_name in {"gsm8k"}:
            self.is_multiple_choice = False
            self.is_multiple_choice_eval = False
        else:
            assert self.dataset_name == "bbh"
            self.is_multiple_choice = []
            self.is_multiple_choice_eval = []
            train_index_by_task_dict = dict()
            eval_index_by_task_dict = dict()
            start_index = 0
            for task_name in self.evolution_config.tasks_all:
                single_task_list = eval_utils.load_bbh_task_data(
                    task_name, base_dir=self.evolution_config.root_data_folder_path
                )
                end_index = start_index + len(single_task_list)
                train_index_by_task_dict[task_name] = (
                    self.train_index[(self.train_index >= start_index) & (self.train_index < end_index)]
                    # if " - start_index" is added here, then the dict would contain
                    # indices in the original task
                )
                eval_index_by_task_dict[task_name] = (
                    self.eval_index[(self.eval_index >= start_index) & (self.eval_index < end_index)]
                    # if " - start_index" is added here, then the dict would contain
                    # indices in the original task
                )
                start_index = end_index
                is_multiple_choice_single_task_train = [
                                                           task_name in self.evolution_config.multiple_choice_tasks
                                                       ] * len(train_index_by_task_dict[task_name])
                is_multiple_choice_single_task_eval = [
                                                          task_name in self.evolution_config.multiple_choice_tasks
                                                      ] * len(eval_index_by_task_dict[task_name])
                self.is_multiple_choice += is_multiple_choice_single_task_train
                self.is_multiple_choice_eval += is_multiple_choice_single_task_eval

        # evaluate initial instructions
        print("\n============== evaluating initial instructions ===============")
        raw_data = self.evolution_config.raw_data
        for instruction in self.evolution_config.initial_instructions:
            print(f"""computing the score of "{instruction}" by prompting""")
            detailed_results_df = eval_utils.evaluate_single_instruction(
                data=raw_data,
                instruction=instruction,
                eval_index_all=self.train_index,
                scorer_llm=self.scorer_llm,
                is_multiple_choice=self.is_multiple_choice,
                prediction_num_decimals=0,
                max_retry=120,
                sleep_time=60,
                **self.evolution_config,
            )

        self.detailed_results_df_by_instruction_dict[instruction] = detailed_results_df
        scores = detailed_results_df["accuracy"]
        average_score = np.average(scores)
        print(f"instruction: {instruction}, score: {average_score}")
        filename = eval_utils.instruction_to_filename(instruction)
        file_path = os.path.join(self.evolution_config.result_by_instruction_folder, f"{filename}.csv")
        detailed_results_df.to_csv(file_path, index=True, header=True)
        print(f"""saving results of "{instruction}" to {file_path}""")
        self.old_instructions_and_scores.append((instruction, average_score, -1))
        self.old_instructions_and_scores_raw.append((instruction, average_score, -1))
        self.instruction_score_dict[instruction] = average_score

        # increment the counter on wrong questions
        wrong_question_indices_set = set(
            list(
                detailed_results_df.iloc[
                np.where(detailed_results_df.accuracy == 0.0)[0], :
                ].index
            )
        )
        for idx in wrong_question_indices_set:
            self.wrong_questions_from_start_counter[idx] += 1

        for i_step in range(self.evolution_config.num_search_steps):
            print(f"\n================== Step {i_step} =====================")
            if not i_step % 10:
                print(f"old_instructions_and_scores: {self.old_instructions_and_scores}")

            self.step(i_step, **self.evolution_config)

    def step(self,
             i_step,
             few_shot_qa_pairs,
             few_shot_selection_criteria,
             num_few_shot_questions_for_instruction_refinement,
             old_instruction_score_threshold,
             max_num_instructions,
             num_score_buckets,
             result_by_instruction_folder,
             include_qa,
             raw_data,
             meta_prompt_instructions_before_exemplars,
             num_generated_instructions_in_each_step,
             evaluate_generated_ins_on_few_shot,
             optimizer_llm_temperature,
             evaluate_old_ins_on_few_shot,
             eval_interval,
             eval_ratio,
             **kwargs,
             ):
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
        if few_shot_qa_pairs:
            if few_shot_selection_criteria == "accumulative_most_frequent":
                # select QA pairs that were done wrong the most number of times
                most_frequent_wrong_question_indices = [
                    k
                    for k, _ in sorted(
                        self.wrong_questions_from_start_counter.items(), key=lambda x: -x[1]
                    )
                ]
                print(
                    "len(most_frequent_wrong_question_indices):"
                    f" {len(most_frequent_wrong_question_indices)}"
                )
                if (
                        len(most_frequent_wrong_question_indices)
                        <= num_few_shot_questions_for_instruction_refinement
                ):
                    few_shot_index_list = most_frequent_wrong_question_indices.copy()
                else:
                    np.random.seed(i_step)
                    few_shot_index_list = np.sort(
                        np.random.choice(
                            most_frequent_wrong_question_indices,
                            num_few_shot_questions_for_instruction_refinement,
                            replace=False,
                        )
                    )

            elif few_shot_selection_criteria == "current_most_frequent":
                # show exemplars done wrong most often by currently shown instructions
                old_instruction_score_threshold_single_step = (
                    old_instruction_score_threshold if i_step > 0 else 0
                )
                _, old_instructions_and_scores_in_meta_prompt = (
                    opt_utils.gen_ins_and_score_pairs_substr(
                        old_instructions_and_scores=self.old_instructions_and_scores,
                        old_instruction_score_threshold=old_instruction_score_threshold_single_step,
                        max_num_instructions=max_num_instructions,
                        return_str_only=False,
                        num_score_buckets=num_score_buckets,
                    )
                )
                wrong_questions_counter_single_step = collections.Counter()
                for ins, _, _ in old_instructions_and_scores_in_meta_prompt:
                    filename = eval_utils.instruction_to_filename(ins)
                    file_path = os.path.join(
                        result_by_instruction_folder, f"{filename}.csv"
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
                        < num_few_shot_questions_for_instruction_refinement
                ):
                    # pylint: disable=cell-var-from-loop
                    idx_most_to_least = sorted(
                        wrong_questions_counter_single_step,
                        key=lambda x: -wrong_questions_counter_single_step[x],
                    )
                    few_shot_index_list = idx_most_to_least[
                                          :num_few_shot_questions_for_instruction_refinement
                                          ]
                else:
                    few_shot_index_list = np.sort(
                        np.random.choice(
                            most_occurred_wrong_questions,
                            num_few_shot_questions_for_instruction_refinement,
                            replace=False,
                        )
                    )
            elif few_shot_selection_criteria == "constant":
                np.random.seed(0)
                few_shot_index_list = np.sort(
                    np.random.choice(
                        self.train_index,
                        num_few_shot_questions_for_instruction_refinement,
                        replace=False,
                    )
                )
            else:
                assert few_shot_selection_criteria == "random"
                np.random.seed(i_step)
                few_shot_index_list = np.sort(
                    np.random.choice(
                        self.train_index,
                        num_few_shot_questions_for_instruction_refinement,
                        replace=False,
                    )
                ).tolist()
            self.few_shot_index_list_by_step_dict[i_step] = few_shot_index_list
            meta_prompt = opt_utils.gen_meta_prompt_with_ph(
                old_instructions_and_scores=self.old_instructions_and_scores,
                instruction_pos=self.instruction_pos,
                optimizer_llm_name=self.optimizer_llm_name,
                old_instruction_score_threshold=old_instruction_score_threshold,
                max_num_instructions=max_num_instructions,
                meta_prompt_type=self.meta_prompt_type,
                few_shot_qa_pairs=few_shot_qa_pairs,
                include_qa=include_qa,
                data=raw_data,
                few_shot_index_list=few_shot_index_list,
                instructions_before_exemplars=meta_prompt_instructions_before_exemplars,
                num_score_buckets=num_score_buckets,
                dataset_name=self.dataset_name,
                task_name=self.task_name,
                prompt_handler=self.prompt_handler,
            )

        else:  # no few-shot exemplars in meta-prompt
            few_shot_index_list = []
            meta_prompt = opt_utils.gen_meta_prompt_with_ph(
                old_instructions_and_scores=self.old_instructions_and_scores,
                instruction_pos=self.instruction_pos,
                optimizer_llm_name=self.optimizer_llm_name,
                old_instruction_score_threshold=old_instruction_score_threshold,
                max_num_instructions=max_num_instructions,
                meta_prompt_type=self.meta_prompt_type,
                few_shot_qa_pairs=False,
                include_qa=include_qa,
                instructions_before_exemplars=meta_prompt_instructions_before_exemplars,
                num_score_buckets=num_score_buckets,
                dataset_name=self.dataset_name,
                task_name=self.task_name,
                prompt_handler=self.prompt_handler
            )
        print(f"\nmeta_prompt: \n\n{meta_prompt}\n")
        self.meta_prompts.append((meta_prompt, i_step))
        remaining_num_instructions_to_generate = (
            num_generated_instructions_in_each_step
        )
        generated_instructions_raw = []

        sync_models = [GenerationModel, OpenAIPostModel, OpenAIGenerationModel]
        async_models = [AioGenerationModel, OpenAIAioPostModel, OpenAIAioGenerationModel]

        if isinstance(self.optim_llm, tuple(sync_models)):
            while remaining_num_instructions_to_generate > 0:
                optimizer_llm_input_text = meta_prompt
                # generate instructions
                # print(f"current temperature: {optimizer_llm_temperature_curr}")
                raw_outputs = self.optim_llm.call(prompt=optimizer_llm_input_text).message.content
                # print(raw_outputs)
        elif isinstance(self.optim_llm, tuple(async_models)):
            import asyncio
            optimizer_llm_input_text = [meta_prompt for _ in range(num_generated_instructions_in_each_step)]
            raw_outputs = [x.message.content for x in asyncio.run(
                self.optim_llm.async_call(prompts=optimizer_llm_input_text, temperature=optimizer_llm_temperature))]
        # import pdb; pdb.set_trace()
        # Extract the generated instructions from the optimizer LLM output. Only
        # keep some samples if the desired number of remaining instructions
        # is smaller than the total number of decodes in this step.
        if self.meta_prompt_type == "both_instructions_and_exemplars":
            if self.optimizer_llm_name.lower() in opt_utils.OPENAI_MODELS:
                raw_outputs = raw_outputs[:remaining_num_instructions_to_generate]
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
                    opt_utils.extract_string_in_square_brackets(string)
                    for string in raw_outputs
                ]
            # remaining_num_instructions_to_generate -= len(raw_outputs)
            remaining_num_instructions_to_generate -= 1
        else:
            assert self.meta_prompt_type == "instructions_only"
            max_num_instructions_to_keep_in_each_output = 1
            for string in raw_outputs:
                generated_instructions_raw += opt_utils.parse_tag_content(string)[
                                              :max_num_instructions_to_keep_in_each_output
                                              ]
            remaining_num_instructions_to_generate -= (
                    len(raw_outputs)
                    * max_num_instructions_to_keep_in_each_output
            )
        # import pdb; pdb.set_trace()
        generated_instructions_raw = list(
            map(eval_utils.polish_sentence, generated_instructions_raw)
        )
        print(f"\ninitially generated instructions: {generated_instructions_raw}\n")
        # do not evaluate old instructions again
        generated_instructions = []  # the new instructions generated in this step
        for ins in generated_instructions_raw:
            ins_md5_hashstring = eval_utils.instruction_to_filename(
                ins, md5_hashing=True
            )
            if ins_md5_hashstring not in self.old_instruction_md5_hashstrings_set:
                generated_instructions.append(ins)
                self.old_instruction_md5_hashstrings_set.add(ins_md5_hashstring)
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
        print(f"\nto-evaluate generated instructions: {to_evaluate_instructions}\n")

        # evaluate new instructions on the few-shot exemplars in meta-prompt
        if few_shot_qa_pairs and evaluate_generated_ins_on_few_shot:
            print("evaluating GENERATED instructions on few-shot exemplars")
            single_step_eval_on_few_shot = dict()
            for instruction in to_evaluate_instructions:
                if instruction not in self.prev_saved_instructions:
                    print(
                        f"evaluating Step {i_step}, instruction: {instruction} on"
                        " few-shot exemplars"
                    )
                    detailed_results_df = eval_utils.evaluate_single_instruction(
                        data=raw_data,
                        instruction=instruction,
                        eval_index_all=few_shot_index_list,
                        scorer_llm=self.scorer_llm,
                        dataset_name=self.dataset_name,
                        include_qa=include_qa,
                        instruction_pos=self.instruction_pos,
                        is_multiple_choice=self.is_multiple_choice,
                        prediction_num_decimals=0,
                        max_retry=5,
                        sleep_time=180,
                        **self.evolution_config,
                    )
                    single_step_eval_on_few_shot[instruction] = detailed_results_df

            print(
                f"Step {i_step}, single_step_eval_on_few_shot:"
                f" {single_step_eval_on_few_shot}\n"
            )
            self.generated_ins_on_few_shot_results_dict[i_step] = (
                single_step_eval_on_few_shot
            )

        # evaluate OLD instructions on the few-shot exemplars in meta-prompt
        if few_shot_qa_pairs and evaluate_old_ins_on_few_shot:
            print("evaluating OLD instructions on few-shot exemplars")
            single_step_eval_on_few_shot = dict()
            for instruction, _, _ in self.old_instructions_and_scores:
                print(
                    f"evaluating Step {i_step}, instruction: {instruction} on few-shot"
                    " exemplars"
                )
                detailed_results_df = eval_utils.evaluate_single_instruction(
                    data=raw_data,
                    instruction=instruction,
                    eval_index_all=few_shot_index_list,
                    scorer_llm=self.scorer_llm,
                    dataset_name=self.dataset_name,
                    include_qa=include_qa,
                    instruction_pos=self.instruction_pos,
                    is_multiple_choice=self.is_multiple_choice,
                    prediction_num_decimals=0,
                    max_retry=5,
                    sleep_time=180,
                    **self.evolution_config,
                )
                single_step_eval_on_few_shot[instruction] = detailed_results_df

            print(
                f"Step {i_step}, single_step_eval_on_few_shot:"
                f" {single_step_eval_on_few_shot}\n"
            )
            self.old_ins_on_few_shot_results_dict[i_step] = single_step_eval_on_few_shot

        # evaluate newly generated instructions on the training set
        for instruction in to_evaluate_instructions:
            if instruction not in self.prev_saved_instructions:
                print(f"""computing the score of "{instruction}" by prompting""")
                detailed_results_df = eval_utils.evaluate_single_instruction(
                    data=raw_data,
                    instruction=instruction,
                    eval_index_all=self.train_index,
                    scorer_llm=self.scorer_llm,
                    include_qa=include_qa,
                    is_multiple_choice=self.is_multiple_choice,
                    prediction_num_decimals=0,
                    max_retry=5,
                    sleep_time=180,
                    **kwargs,
                )
                self.prev_saved_instructions.add(instruction)
            else:
                # do not re-evaluate instructions that had been evaluated previously
                detailed_results_df = pd.read_csv(
                    os.path.join(result_by_instruction_folder, f"{instruction}.csv"),
                    index_col=0,
                    header=0,
                )
                print(f"""reading previously saved "{instruction}" information""")

            scores = detailed_results_df["accuracy"]
            average_score = np.average(scores)
            print(
                f"Step {i_step}, instruction: {instruction}, score: {average_score}"
            )

            # increment the counter on wrong questions
            wrong_question_indices_set = set(
                list(
                    detailed_results_df[detailed_results_df["accuracy"] == 0.0].index
                )
            )
            for idx in wrong_question_indices_set:
                self.wrong_questions_from_start_counter[idx] += 1

            filename = eval_utils.instruction_to_filename(instruction)
            file_path = os.path.join(
                result_by_instruction_folder, f"""{filename}.csv"""
            )
            detailed_results_df.to_csv(file_path, index=True, header=True)
            print(f"saving results to {file_path}")

            self.detailed_results_df_by_instruction_dict[instruction] = detailed_results_df
            self.old_instructions_and_scores.append((instruction, average_score, i_step))
            self.instruction_score_dict[instruction] = average_score

        # record all generated instructions
        for instruction in generated_instructions_raw:
            if instruction in self.instruction_score_dict:
                average_score = self.instruction_score_dict[instruction]
            else:
                average_score = np.nan
                self.old_instructions_and_scores_raw.append(
                    (instruction, average_score, i_step)
                )

        # =============================== eval ====================================
        # every eval_interval steps, evaluate the instructions that were generated
        # in the current step and were not skipped
        if not i_step % eval_interval and eval_ratio > 0:
            for instruction in generated_instructions_raw:
                # if the instruction wasn't skipped in any step
                if instruction in self.scorer_llminstruction_score_dict:
                    if instruction not in self.instruction_eval_score_dict:
                        detailed_results_df = eval_utils.evaluate_single_instruction(
                            data=raw_data,
                            instruction=instruction,
                            eval_index_all=self.eval_index,
                            scorer_llm=self.scorer_llm,
                            include_qa=include_qa,
                            is_multiple_choice=self.is_multiple_choice_eval,
                            prediction_num_decimals=0,
                            max_retry=5,
                            sleep_time=180,
                            **kwargs,
                        )
                        eval_score = np.average(detailed_results_df["accuracy"])
                        self.eval_detailed_results_df_dict[instruction] = detailed_results_df
                        self.instruction_eval_score_dict[instruction] = eval_score
                    else:
                        eval_score = self.instruction_eval_score_dict[instruction]
                    print(
                        f"EVAL: \nStep {i_step}, instruction: {instruction}, eval score:"
                        f" {eval_score:.2f}"
                    )
                    self.eval_results.append((i_step, instruction, eval_score))

        # ===================== save up-to-date results ===========================
        results_dict = dict()
        results_dict["meta_prompts"] = self.meta_prompts
        results_dict["old_instructions_and_scores"] = list(
            self.old_instructions_and_scores
        )
        results_dict["old_instructions_and_scores_raw"] = list(
            self.old_instructions_and_scores_raw
        )
        results_dict["generated_ins_on_few_shot_results_dict"] = (
            self.generated_ins_on_few_shot_results_dict
        )
        results_dict["old_ins_on_few_shot_results_dict"] = (
            self.old_ins_on_few_shot_results_dict
        )
        results_dict["few_shot_index_list_by_step_dict"] = (
            self.few_shot_index_list_by_step_dict
        )
        results_dict["eval_results"] = self.eval_results
        results_dict["eval_detailed_results_df_dict"] = (
            self.eval_detailed_results_df_dict
        )

        import pickle
        save_folder = kwargs.get('save_folder')
        with open(os.path.join(save_folder, "results_dict.pkl"), "wb") as fp:
            pickle.dump(results_dict, fp)
        print(f"\nsaved all results to\n{save_folder}")

    def extract_best_prompt(self):
        # todo: by jm, return best prompt to align the interface definition.
        """
		This method is intended to extract the best prompt from the evaluated results.
		Currently, it is a placeholder and needs implementation.
		"""
        pass
