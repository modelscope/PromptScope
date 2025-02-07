import json
import os
import random

from tqdm.asyncio import tqdm
from pathlib import Path
from loguru import logger
from typing import List, Dict, Any, Callable, Tuple
import asyncio
import nest_asyncio
nest_asyncio.apply()

from prompt_scope.core.utils.file_utils import save_eval_details
from prompt_scope.core.schemas.example import LLMCallRecord
from prompt_scope.core.prompt_gen.prompt_gen import BasePromptGen
from prompt_scope.core.llms.base import BaseLLM
from prompt_scope.core.optimizer.tips_optimizer.utils import evaluate_on_dataset, evaluate_on_sample
from .base_tips_optimizer import BaseTipsOptimizer

current_file_dir = Path(__file__).parent


class StaticTipsOptimizer(BaseTipsOptimizer):
    def __init__(self,
                 infer_llm: BaseLLM,
                 optim_llm: BaseLLM,
                 init_system_prompt: str = None,
                 train_set: List[Dict] = None,
                 test_set: List[Dict] = None,
                 bad_case_analysis_prompt_dir: str = None,
                 details_save_dir: str = None,
                 is_good_case_func: Callable[[str, str], bool] = None,
                 resume_generate: bool = True,
                 language: str = "cn",
                 epoch: int = 1,
                 train_bach_size: int = 1,
                 save_steps: int = 50,
                 early_stopping: bool = False,
                 max_workers_num: int = 20
                 ):
        super().__init__(infer_llm=infer_llm,
                         optim_llm=optim_llm,
                         init_system_prompt=init_system_prompt,
                         train_set=train_set,
                         test_set=test_set,
                         bad_case_analysis_prompt_dir=bad_case_analysis_prompt_dir,
                         details_save_dir=details_save_dir,
                         is_good_case_func=is_good_case_func,
                         resume_generate=resume_generate,
                         language=language,
                         epoch=epoch,
                         train_bach_size=train_bach_size,
                         save_steps=save_steps,
                         max_workers_num=max_workers_num)
        self.early_stopping = early_stopping

        self._init_optimizer_config()

    def _init_optimizer_config(self):
        super()._init_optimizer_config()

        # if self.summarize_prompt_dir is None:
        #     self.summarize_prompt_dir = \
        #         f"{current_file_dir}/prompt_lib/{self.language}/summarize_prompt_{self.language}"
        #
        # self.summarize_prompt = BasePromptGen.load(promptgen_load_dir=self.summarize_prompt_dir)

    def _before_run(self) -> Dict[str, Any]:
        if self.init_system_prompt is not None:
            for sample in self.train_set:
                sample.system_prompt = self.init_system_prompt
            for sample in self.test_set:
                sample.system_prompt = self.init_system_prompt


        # evaluate initial instructions on test
        logger.debug("\n============== evaluating initial system prompt on test ===============")
        test_avg_score, test_result = evaluate_on_dataset(
            infer_llm=self.infer_llm,
            dataset=self.test_set,
            is_good_case_func=self.is_good_case_func,
            semaphore=self.semaphore
        )
        logger.debug(f"test_avg_score: \n{test_avg_score}")
        if self.details_save_dir is not None:
            saved_prompt = f"{self.init_system_prompt}"
            self._save_details(saved_prompt=saved_prompt, detailed_data=test_result, save_dir=f"{self.details_save_dir}/init")

        return test_avg_score, test_result

    def run(self,) -> Dict[str, Any]:
        all_test_detailed_result = []

        init_test_avg_score, init_test_result = self._before_run()
        all_test_detailed_result.append(init_test_avg_score)

        i_step = 0
        train_records = [("init", init_test_avg_score, self.init_system_prompt, init_test_result)]
        latest_tips = None
        total_steps = len(self.train_set)//self.train_bach_size
        if len(self.train_set) % self.train_bach_size > 0:
            total_steps += 1

        bad_case_list = []
        total_steps = total_steps*self.epoch
        for i_epoch in range(1, self.epoch+1):
            for start_idx in tqdm(range(0, len(self.train_set), self.train_bach_size),
                                  desc=f"Epoch-{i_epoch}/{self.epoch} Training", ncols=80):
                i_step += 1

                batched_train = self.train_set[start_idx: (start_idx+self.train_bach_size)]

                bad_case_feedbacks, add_tips = self._step(batched_train, latest_tips)

                if bad_case_feedbacks is not None and len(bad_case_feedbacks) > 0:
                    bad_case_list.extend(bad_case_feedbacks)

                latest_tips = self._merge_tips(latest_tips, add_tips)

                if i_step % self.save_steps == 0 or i_step == total_steps:
                    logger.debug(f"\n=====Epoch-{i_epoch} Step-{i_step}/{total_steps}=====")
                    logger.debug(f"\n=====Epoch-{i_epoch}/{self.epoch} Step-{i_step}/{total_steps} updated tips=====\n{latest_tips}")

                    saved_prompt = f"{self.init_system_prompt}\n\n{latest_tips}"

                    if saved_prompt != train_records[-1][2]:
                        logger.debug(f"\n=====Epoch-{i_epoch}/{self.epoch} Step-{i_step}/{total_steps} evaluate updated tips=====\n")
                        test_avg_score, test_result = evaluate_on_dataset(self.infer_llm, self.test_set,
                                                                          self.is_good_case_func,
                                                                          tips=latest_tips, semaphore=self.semaphore)
                    else:
                        test_avg_score = train_records[-1][1]
                        test_result = train_records[-1][3]

                    logger.debug(f"\ntest_avg_score: {test_avg_score}")

                    train_records.append((f"{i_step}", test_avg_score, saved_prompt, test_result))

                    self._save_details(saved_prompt=saved_prompt, detailed_data=test_result,
                                       save_dir=f"{self.details_save_dir}/step-{i_step}", bad_case_feedbacks=bad_case_list)
                    bad_case_list = []


                    best_test_score = -1
                    best_step = None
                    best_prompt = self.init_system_prompt
                    with open(f"{self.details_save_dir}/test_details.txt", "w") as f:
                        for item in train_records:
                            tmp_step = item[0]
                            tmp_test_score = item[1]
                            tmp_prompt = item[2]
                            f.write(f"step-{tmp_step}: {tmp_test_score}\n")
                            if tmp_test_score > best_test_score:
                                best_step = tmp_step
                                best_test_score = tmp_test_score
                                best_prompt = tmp_prompt

                        f.write(f"best step-{best_step}: {best_test_score}")

                    with open(f"{self.details_save_dir}/best_prompt.txt", "w") as f:
                        f.write(best_prompt)

                    if test_avg_score == 1:
                        break

                    if self.early_stopping and test_avg_score < best_test_score:
                        break

        return best_prompt

    def _step(self, batched_train: List[LLMCallRecord], latest_tips: str):
        # analysis bad cases
        chosen_latest_tips = None
        if latest_tips is not None:
            latest_tips_list = [tip.strip() for tip in latest_tips.split("\n") if len(tip.strip())>0]
            random.shuffle(latest_tips_list)
            chosen_latest_tips = "\n".join(latest_tips_list[:5])
        bad_case_feedbacks = self._evaluate_and_analyze(batched_train, chosen_latest_tips)

        # generate tips
        add_tips = self._update_tips(bad_case_feedbacks=bad_case_feedbacks)

        return bad_case_feedbacks, add_tips

    def _evaluate_and_analyze(self, batched_train: List[LLMCallRecord], latest_tips: str):
        train_avg_score, train_results = evaluate_on_dataset(self.infer_llm, batched_train, self.is_good_case_func,
                                                             latest_tips, semaphore=self.semaphore, show_process_bar=False)
        bad_cases = self._split_data(train_results)

        if len(bad_cases) > 0:
            bad_case_feedbacks = asyncio.run(self._generate_analysis(bad_cases=bad_cases, tips=latest_tips))

            return bad_case_feedbacks
        else:
            return None

    def _split_data(self, dataset: List[LLMCallRecord]):
        bad_cases = []
        for sample in dataset:
            if not sample.is_good_case:
                bad_cases.append(sample)
        return bad_cases

    async def _generate_analysis(self, bad_cases: List[LLMCallRecord], tips: str):
        tasks = [self._async_generate_analysis_on_case(bc, tips) for bc in bad_cases]

        bad_case_feedbacks = []
        for result in tasks:
            bad_case, bc_feedback = await result
            if bc_feedback is not None:
                bad_case_feedbacks.append((bad_case, bc_feedback))

        return bad_case_feedbacks

    async def _async_generate_analysis_on_case(self, bad_case: LLMCallRecord, tips: str):
        async with self.semaphore:
            return await asyncio.to_thread(self._generate_analysis_on_case, bad_case, tips)

    def _generate_analysis_on_case(self, bad_case: LLMCallRecord, tips: str):
        final_bc_feedback = None

        if tips is not None and len(tips.strip()) > 0:
            tips_list = [t.strip() for t in tips.split("\n") if len(t.strip()) > 0]
            random.shuffle(tips_list)
            sample_tips_str = "\n".join(tips_list[:5])
        else:
            sample_tips_str = ""

        llm_input = {
            "${system_prompt}": f"{bad_case.system_prompt}\n\n{sample_tips_str}",
            "${query}": bad_case.input,
            "${prediction}": bad_case.prediction,
            "${ground_truth}": bad_case.output,
        }

        bc_feedback = self.bad_case_analysis_prompt.generate(llm=self.optim_llm, llm_input=llm_input)
        # print(bc_feedback)
        bc_feedback = self._extract_tips(bc_feedback)

        if bc_feedback is not None:
            bad_case_copy = bad_case.model_copy(deep=True)
            bad_case_copy.tips = bc_feedback
            tmp_score = evaluate_on_sample(self.infer_llm, bad_case_copy, self.is_good_case_func)

            if tmp_score > bad_case.score:
                # print(bc_feedback)
                final_bc_feedback = bc_feedback
            # else:
            #     print(tmp_score)
            #     print(bad_case_copy)
            #     input()

        return bad_case, final_bc_feedback

    def _update_tips(self, bad_case_feedbacks: List[Tuple[LLMCallRecord, str]]):
        if bad_case_feedbacks is not None and len(bad_case_feedbacks) > 0:
            feedbacks = [fb[-1] for fb in bad_case_feedbacks]
            feedback_all = "\n".join(feedbacks)
            # tips = self.summarize_prompt.generate(llm_input={"${feedback_all}": feedback_all})
            tips = feedback_all
            return tips
        else:
            return None

    def _merge_tips(self, latest_tips, add_tips):
        # if add_tips is not None and len(add_tips) > 0:
        #     if latest_tips is not None:
        #         latest_tips = f"{latest_tips}\n{add_tips}"
        #     else:
        #         latest_tips = add_tips

        # if add_tips is not None and len(add_tips) > 0:
        #     if latest_tips is not None:
        #         latest_tips = self.summarize_prompt.generate(
        #             llm=self.optim_llm,
        #             llm_input={
        #                 "${latest_tips}": f"{latest_tips}",
        #                 "${add_tips}": f"{add_tips}"
        #             })
        #
        #         latest_tips = self._extract_tips(latest_tips)
        #         if latest_tips is None or len(latest_tips.strip())==0:
        #             latest_tips = f"{latest_tips}\n\n{add_tips}"
        #     else:
        #         latest_tips = add_tips

        if add_tips is not None and len(add_tips) > 0:
            if latest_tips is not None:
                add_tips_list = [tip.strip() for tip in add_tips.split("\n") if len(tip.strip())>0 and tip not in latest_tips]
                add_tips_filter = "\n".join(add_tips_list)
                if len(add_tips_filter) > 0:
                    latest_tips = f"{latest_tips}\n\n{add_tips_filter}"
            else:
                latest_tips = add_tips

        return latest_tips

    def extract_best_prompt(self, history_scores, history_tips):
        best_tips = history_tips[0]
        best_score = history_scores[0]
        for score, tips in zip(history_scores, history_tips):
            if score >= best_score:
                best_score = score
                best_tips = tips

        if self.details_save_dir is not None:
            best_prompt_save_path = f"{self.details_save_dir}/best_tips.txt"
            logger.debug(f"\n=====save best prompt=====\n{best_prompt_save_path}")
            with open(best_prompt_save_path, "w") as f:
                f.write(best_tips)
        return best_tips

    def _save_details(self, saved_prompt: str, detailed_data: List[LLMCallRecord], save_dir: str, bad_case_feedbacks=None):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = f"{save_dir}/test_details.xlsx"
        detailed_data = [json.loads(d.model_dump_json()) for d in detailed_data]
        save_eval_details(detailed_data, save_path)

        with open(f"{save_dir}/saved_prompt.txt", "w") as f:
            f.write(saved_prompt)

        if bad_case_feedbacks is not None:
            bad_case_feedbacks = [{"llm_record": json.loads(d[0].model_dump_json()), "feedback": d[1]}
                                  for d in bad_case_feedbacks]
            with open(f"{save_dir}/bad_case_feedbacks.json", "w") as f:
                json.dump(bad_case_feedbacks, f, ensure_ascii=False, indent=4)
