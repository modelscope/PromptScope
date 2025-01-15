import os
import re
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import List, Dict, Any, Callable

from prompt_scope.core.schemas.example import LLMCallRecord
from prompt_scope.core.prompt_gen.prompt_gen import BasePromptGen
from prompt_scope.core.llms.dashscope_llm import DashscopeLLM
from .base_tips_optimizer import BaseTipsOptimizer
from prompt_scope.core.optimizer.tips_optimizer.utils import evaluate_on_dataset, predict_on_sample

current_file_dir = Path(__file__).parent


class DynamicTipsOptimizer(BaseTipsOptimizer):
    def __init__(self,
                 infer_llm: DashscopeLLM,
                 init_system_prompt: str = None,
                 train_set: List[LLMCallRecord] = None,
                 test_set: List[LLMCallRecord] = None,
                 good_case_analysis_prompt_dir: str = None,
                 bad_case_analysis_prompt_dir: str = None,
                 tips_postprocess_prompt_dir: str = None,
                 details_save_dir: str = None,
                 is_correct_func: Callable[[str, str], bool] = None,
                 resume_generate: bool = True,
                 language: str = "cn",
                 max_iteration: int = 3
                 ):
        super().__init__(
            infer_llm=infer_llm,
            init_system_prompt=init_system_prompt,
            train_set=train_set,
            test_set=test_set,
            good_case_analysis_prompt_dir=good_case_analysis_prompt_dir,
            bad_case_analysis_prompt_dir=bad_case_analysis_prompt_dir,
            details_save_dir=details_save_dir,
            is_correct_func=is_correct_func,
            resume_generate=resume_generate,
            language=language,
            max_iteration=max_iteration

        )
        self.tips_postprocess_prompt = None
        self.tips_postprocess_prompt_dir = tips_postprocess_prompt_dir

        self.details_save_dir = details_save_dir

        self.save_query2tips_path = f"{self.details_save_dir}/query2tips.json"
        self.save_details_path = f"{self.details_save_dir}/train_eval_details.xlsx"

        self.good_cases = []
        self.bad_cases = []

        self._init_optimizer_config()

    def _init_optimizer_config(self):
        super()._init_optimizer_config()

        if self.tips_postprocess_prompt_dir is not None:
            self.tips_postprocess_prompt = BasePromptGen.load(promptgen_load_dir=self.tips_postprocess_prompt_dir)

        self.select_tips_prompt = BasePromptGen.load(
            promptgen_load_dir=f"{current_file_dir}/prompt_lib/{self.language}/select_tips_prompt_{self.language}")

        if os.path.exists(self.save_query2tips_path) and not self.resume_generate:
            os.remove(self.save_query2tips_path)

        if os.path.exists(self.save_details_path) and not self.resume_generate:
            os.remove(self.save_details_path)

    def _before_run(self):
        logger.debug(f"init prediction & evaluation on train")
        if self.init_system_prompt is not None:
            for sample in self.train_set:
                sample.system_prompt = self.init_system_prompt

        train_avg_score, train_result = evaluate_on_dataset(
            infer_llm=self.infer_llm,
            dataset=self.train_set,
            is_correct=self.is_correct_func
        )

        for case in tqdm(train_result):
            if case.is_correct:
                self.good_cases.append(case)
            else:
                self.bad_cases.append(case)

    def _generate_tips(self, case_type: str):
        if case_type == "good":
            case_analysis_prompt = self.good_case_analysis_prompt
            cases = self.good_cases
        elif case_type == "bad":
            case_analysis_prompt = self.bad_case_analysis_prompt
            cases = self.bad_cases
        else:
            raise ValueError(f"case_type must be 'good' or 'bad', but got {case_type}")

        if cases is None or len(cases) == 0:
            return {}

        cases_num = len(cases)
        optimized_prediction_list = []
        for idx, example in tqdm(enumerate(cases)):
            system_prompt = None
            if self.init_system_prompt is not None:
                system_prompt = self.init_system_prompt
            elif example.system_prompt is not None:
                system_prompt = example.system_prompt

            last_user_content = example.last_user_content
            old_prediction = example.prediction
            old_is_correct = example.is_correct
            ground_truth = example.ground_truth
            history = example.history

            if os.path.exists(self.save_query2tips_path):
                with open(self.save_query2tips_path, "r") as f:
                    query2tips = json.load(f)
            else:
                query2tips = {}

            if self.resume_generate and last_user_content in query2tips:
                continue

            if os.path.exists(self.save_details_path):
                details = pd.read_excel(self.save_details_path)
                save_system_prompt_list = details["system_prompt"].tolist()
                save_history_list = details["history"].tolist()
                save_last_user_content_list = details["query"].tolist()
                save_tips_list = details["tips"].tolist()
                save_old_pred_list = details["old_prediction"].tolist()
                save_old_is_correct_list = details["eval_result_for_old_prediction"].tolist()
                save_new_pred_list = details["new_prediction"].tolist()
                save_new_is_correct_list = details["eval_result_for_new_prediction"].tolist()
                save_gt_list = details["gt"].tolist()
            else:
                save_system_prompt_list = []
                save_history_list = []
                save_last_user_content_list = []
                save_tips_list = []
                save_old_pred_list = []
                save_old_is_correct_list = []
                save_new_pred_list = []
                save_new_is_correct_list = []
                save_gt_list = []

            for i_iteration in range(1, self.max_iteration + 1):
                llm_input = {
                    "${system_prompt}": system_prompt,
                    "${query}": last_user_content,
                    "${ground_truth}": ground_truth
                }
                if case_type == "bad":
                    llm_input["${prediction}"] = old_prediction

                if history is not None and len(history) > 0:
                    llm_input["${history}"] = "\n".join([cov_round.text for cov_round in history])

                logger.debug(
                    f"{case_type}case {idx + 1}/{cases_num}, iteration {i_iteration}/{self.max_iteration}, {case_type}case analysis")
                case_analysis = case_analysis_prompt.generate(llm_input=llm_input)
                print(case_analysis)
                case_analysis = self._extract_tips(case_analysis)
                logger.debug(f"case_analysis:\n{case_analysis}")
                if self.tips_postprocess_prompt is not None:
                    case_analysis = self.tips_postprocess_prompt.generate(llm_input={"${case_analysis}": case_analysis})
                generated_tips = self._extract_tips(case_analysis)
                logger.debug(f"generated tips:\n{generated_tips}")

                update_system_prompt = f"{system_prompt}\n\n{generated_tips}"
                logger.debug(
                    f"{case_type}case {idx + 1}/{cases_num}, iteration {i_iteration}/{self.max_iteration}, infer with generated tips")

                new_prediction = predict_on_sample(infer_llm=self.infer_llm, system_prompt=update_system_prompt,
                                                   last_user_content=last_user_content,  history=history)
                logger.debug(f"add tips response:\n{new_prediction}")
                new_is_correct = self.is_correct_func(new_prediction, ground_truth)
                if new_is_correct:
                    break

                # TODO: 前后tips的合并

            optimized_prediction_list.append(new_prediction)

            logger.debug(f"save {case_type}case {idx + 1}/{cases_num} generated tips to {self.save_query2tips_path}")

            query2tips[last_user_content] = generated_tips

            with open(self.save_query2tips_path, "w") as f:
                json.dump(query2tips, f, ensure_ascii=False, indent=4)

            logger.debug(f"add {case_type}case {idx + 1}/{cases_num} details to {self.save_details_path}")

            save_system_prompt_list.append(system_prompt)
            if history is not None:
                save_history_list.append([cov_round.text for cov_round in history])
            else:
                save_history_list.append(None)
            save_last_user_content_list.append(last_user_content)
            save_tips_list.append(generated_tips)
            save_old_pred_list.append(old_prediction)
            save_old_is_correct_list.append(int(old_is_correct))
            save_new_pred_list.append(new_prediction)
            save_new_is_correct_list.append(int(new_is_correct))
            save_gt_list.append(ground_truth)

            pd.DataFrame(
                {
                    "system_prompt": save_system_prompt_list,
                    "history": save_history_list,
                    "query": save_last_user_content_list,
                    "tips": save_tips_list,
                    "old_prediction": save_old_pred_list,
                    "eval_result_for_old_prediction": save_old_is_correct_list,
                    "new_prediction": save_new_pred_list,
                    "eval_result_for_new_prediction": save_new_is_correct_list,
                    "gt": save_gt_list,
                }
            ).to_excel(self.save_details_path)

        return query2tips

    def run(self, ) -> Dict[str, Any]:
        self._before_run()

        good_cases_query2tips = self._generate_tips("good")
        bad_cases_query2tips = self._generate_tips("bad")

        query2tips = {**good_cases_query2tips, **bad_cases_query2tips}

        return query2tips


    def _select_tips(self, ground_truth, tips_list, prediction_list):
        if len(tips_list) == 1:
            return tips_list[0], prediction_list[0]

        idx_prediction_list = []
        for idx, prediction in enumerate(prediction_list):
            idx_prediction_list.append(f"[{idx + 1}]\n{prediction}")

        idx_prediction_str = "\n\n".join(idx_prediction_list)

        llm_input = {
            "${idx_prediction}": idx_prediction_str,
            "${ground_truth}": ground_truth,
        }

        response = self.select_tips_prompt.generate(llm_input=llm_input)
        pattern = r"最相似的句子：\[(\d+)\]"
        matches = re.findall(pattern, response)
        if len(matches) == 0:
            return tips_list[0], prediction_list[0]
        else:
            return tips_list[int(matches[0]) - 1], prediction_list[int(matches[0]) - 1]
