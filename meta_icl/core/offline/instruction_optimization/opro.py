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
import os
import numpy as np
import collections
import pandas as pd

from loguru import logger
from meta_icl.core.models.generation_model import AioGenerationModel, GenerationModel
from meta_icl import CONFIG_REGISTRY
from meta_icl.algorithm.base_algorithm import PromptOptimizationWithFeedback
from meta_icl.algorithm.opro.evaluation import eval_utils
from meta_icl.algorithm.opro.optimization import opt_utils

QWEN_MODELS = {"qwen-turbo",
				"qwen2-57b-a14b-instruct",
				"qwen2-72b-instruct",
				"qwen-max-allinone",
				}

class OPRO(PromptOptimizationWithFeedback):
	FILE_PATH: str = __file__
	def __init__(self, language, **kwargs):
		super().__init__(language, **kwargs)
		self.init_config()
		self.init_model()
		self.scorer_llm_name = self.model_config.scorer.model_name
		self.optimizer_llm_name = self.model_config.optim.model_name
		self.dataset_name = self.task_config.dataset_name.lower()
		self.task_name = self.basic_config.task_name
		self.meta_prompt_type = self.basic_config.meta_prompt_type
		self.instruction_pos = self.basic_config.instruction_pos
		self.validation()
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
		if self.evolution_config.evaluate_in_parallel:
			self.scorer_llm = AioGenerationModel(**CONFIG_REGISTRY.module_dict['model_config'].scorer)
			self.optim_llm = AioGenerationModel(**CONFIG_REGISTRY.module_dict['model_config'].optim)
		else:
			self.scorer_llm = GenerationModel(**CONFIG_REGISTRY.module_dict['model_config'].scorer)
			self.optim_llm = GenerationModel(**CONFIG_REGISTRY.module_dict['model_config'].optim)

	def init_config(self):
		self.model_config = CONFIG_REGISTRY.module_dict['model_config']
		self.task_config = CONFIG_REGISTRY.module_dict['task_config']
		self.basic_config = CONFIG_REGISTRY.module_dict['basic_config']
		self.evolution_config = CONFIG_REGISTRY.module_dict["evolution_config"]

	def update_config(self, config='evolution_config', **kwargs):
		self.__getattribute__(config).update(**kwargs)
	def validation(self):

		assert self.dataset_name in {
			"mmlu",
			"bbh",
			"gsm8k",
		}, "The lower-case dataset name must be one of mmlu, bbh, or gsm8k."
		if self.dataset_name == "mmlu":
			assert self.task_name in {
				"STEM",
				"humanities",
				"social sciences",
				"other (business, health, misc.)",
			}  # for now only support searching on one MMLU category
		elif self.dataset_name == "bbh":
			assert self.task_name in {
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
		else:
			assert self.dataset_name == "gsm8k"
			assert self.task_name in {"train", "test"}

		assert self.scorer_llm_name in QWEN_MODELS
		assert self.optimizer_llm_name in QWEN_MODELS

		assert self.meta_prompt_type in {
			"both_instructions_and_exemplars",
			"instructions_only",
		}

		assert self.instruction_pos in {
			"before_Q",
			"Q_begin",
			"Q_end",
			"A_begin",
		}, (
			"The instruction position should be either before the question, or at the"
			" beginning of the question, at the end of the question, or at the"
			" beginning of the answer."
		)
		print(
			f"scorer: {self.scorer_llm_name}, optimizer: {self.optimizer_llm_name}, dataset:"
			f" {self.dataset_name}, task: {self.task_name}, instruction_pos: {self.instruction_pos}"
		)
	def init_prompt(self):
		pass

	def run(self):
		"""The function for evolution."""
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

		from meta_icl.core.models.generation_model import GenerationModel, AioGenerationModel
		if isinstance(self.optim_llm, GenerationModel):
			while remaining_num_instructions_to_generate > 0:
				optimizer_llm_input_text = meta_prompt
				# generate instructions
				# print(f"current temperature: {optimizer_llm_temperature_curr}")
				raw_outputs = self.optim_llm.call(prompt=optimizer_llm_input_text).message.content
				# print(raw_outputs)
			# Extract the generated instructions from the optimizer LLM output. Only
			# keep some samples if the desired number of remaining instructions
			# is smaller than the total number of decodes in this step.
				if self.meta_prompt_type == "both_instructions_and_exemplars":
					if self.optimizer_llm_name.lower() in {"gpt-3.5-turbo", "gpt-4"}:
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
						assert self.optimizer_llm_name.lower() in QWEN_MODELS
						generated_instructions_raw += [
							opt_utils.extract_string_in_square_brackets(string)
							for string in raw_outputs
						]
						generated_instructions_raw.append(raw_outputs[1:-1])

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
		elif isinstance(self.optim_llm, AioGenerationModel):
			import asyncio
			optimizer_llm_input_text = [meta_prompt for _ in range(num_generated_instructions_in_each_step)]
			raw_outputs = [x.message.content for x in asyncio.run(self.optim_llm.async_call(prompts=optimizer_llm_input_text, temperature=optimizer_llm_temperature))]
			generated_instructions_raw = [
							opt_utils.extract_string_in_square_brackets(string)[1:-1]
							for string in raw_outputs
						]
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
		pass