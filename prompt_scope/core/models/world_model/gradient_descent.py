# The code is modified based on Automatic Prompt Optimization with "Gradient Descent" and Beam Search
# https://arxiv.org/abs/2305.03495

import re
from loguru import logger
import numpy as np
from typing import Dict, Any

from prompt_scope.core.utils.prompt_handler import PromptHandler
from prompt_scope.core.llms.base import BaseLLM


class GradientDescent():
    def __init__(self,
                 task,
                 base_model: BaseLLM,
                 optim_model: BaseLLM,
                 print_log=True,
                 logger=None,
                 num_new_prompts=1,
                 prompt_length_limit=200,
                 prompt_handler: PromptHandler = None, ):
        self.task = task
        self.base_model = base_model
        self.optim_model = optim_model
        self.print_log = print_log if logger is not None else False
        self.num_new_prompts = num_new_prompts

        self.use_correct_examples = False

        self.optimize_prompt_template = prompt_handler.optimize_prompt_template_single \
            if num_new_prompts == 1 else prompt_handler.optimize_prompt_template
        self.ascend_optimize_prompt_template = prompt_handler.ascend_optimize_prompt_template_single \
            if num_new_prompts == 1 else prompt_handler.ascend_optimize_prompt_template
        self.gradient_prompt_template = prompt_handler.gradient_prompt_template
        self.ascend_gradient_prompt_template = prompt_handler.ascend_gradient_prompt_template
        self.example_template = prompt_handler.example_template

        self.forward_log_template = prompt_handler.forward_log_template
        self.gradient_log_template = prompt_handler.gradient_log_template
        self.optimize_log_template = prompt_handler.optimize_log_template

        self._build_forward_prompts_func = task.build_forward_prompts_completion
        self.call_func = self.base_model.chat

    def forward(self, batch, cur_prompt):
        batch_size = len(batch['question'])
        batch_prompts = self._build_forward_prompts_func(batch['question'], cur_prompt)
        try:
            responses = [self.call_func(messages=prompt).message.content for prompt in batch_prompts]
        except Exception:
            responses = [self.call_func(messages=prompt).output.text for prompt in batch_prompts]

        for p, r in zip(batch_prompts, responses):
            logger.info(f"Input:\n{p}")
            logger.info(f"Output:\n{r}")

        preds = self.task.batch_clean_responses(responses)

        labels = self.task.clean_labels(batch['answer'])
        correct = self.task.cal_correct(preds, labels)

        batch_logs = []
        for i in range(batch_size):
            batch_logs.append({
                'cur_prompt': cur_prompt,
                'question': batch['question'][i],
                'model_input': batch_prompts[i],
                'gt_answer': batch['answer'][i],
                'model_response': responses[i],
                'label': labels[i],
                'pred': preds[i],
            })

        forward_output = {
            'cur_prompt': cur_prompt,
            'correct': correct,
            'examples': batch_logs,
            'acc': np.mean(correct)
        }

        if self.print_log:
            log_str = self.forward_log_template.format(
                cur_prompt=cur_prompt,
                batch_prompts=batch_prompts,
                responses=responses,
                preds=preds,
                labels=labels,
                correct=forward_output['correct'],
                acc=forward_output['acc'])

            logger.info(log_str)
        return forward_output

    def _clean_self_eval_score(self, response):
        return re.findall(r'\d+', response)[-1]

    def _split_error_and_correct_examples(self, forward_output):
        error_examples = []
        correct_examples = []
        count = 0
        for i, example in enumerate(forward_output['examples']):
            if forward_output['correct'][i] == 0:
                count += 1
                error_examples.append(self.example_template.format(
                    index=count,
                    question=example['model_input'],
                    label=example['label'],
                    response=example['model_response'],
                    prediction=example['pred']))
            elif forward_output['correct'][i] == 1:
                count += 1
                correct_examples.append(self.example_template.format(
                    index=count,
                    question=example['model_input'],
                    label=example['label'],
                    response=example['model_response'],
                    prediction=example['pred']))
            else:
                raise ValueError(f'_get_error_examples: invalid correct number {i} {forward_output}.')
        error_string = ''.join(error_examples)
        correct_string = ''.join(correct_examples)
        return error_string, correct_string

    def _build_prompt_trajectory_str(self, prompts):
        prompt_path_str = ""
        prompt_path_str_template = "({index}) {prompt}\n"
        for i, prompt in enumerate(prompts):
            prompt_path_str += prompt_path_str_template.format(index=i, prompt=prompt)
        return prompt_path_str

    def cal_gradient(self, cur_prompt, example_string, gradient_prompt_template):
        gradient_prompt = gradient_prompt_template.format(cur_prompt=cur_prompt,
                                                          example_string=example_string)
        
        gradient = self.optim_model.chat(messages=gradient_prompt).message.content
        if self.print_log:
            log_str = self.gradient_log_template.format(gradient_prompt=gradient_prompt,
                                                        gradient=gradient)
            logger.info(log_str)

        return gradient

    def _clean_optim_response(self, optim_response):
        pattern = r'<START>(.*?)<END>'
        matches = re.findall(pattern=pattern, string=optim_response, flags=re.DOTALL)
        for i, m in enumerate(matches):
            matches[i] = m.strip()
        return matches

    def optimize(self, cur_prompt, example_string, gradient, trajectory_prompts,
                 steps_per_gradient, optimize_prompt_template):
        optimize_prompt = optimize_prompt_template.format(
            cur_prompt=cur_prompt,
            example_string=example_string,
            gradient=gradient,
            trajectory_prompts=trajectory_prompts,
            steps_per_gradient=steps_per_gradient)
        
        response = self.optim_model.chat(messages=optimize_prompt).message.content
        optimized_prompt = self._clean_optim_response(response)
        if self.print_log:
            log_str = self.optimize_log_template.format(optimize_prompt=optimize_prompt,
                                                        response=response,
                                                        optimized_prompt=optimized_prompt)
            logger.info(log_str)

        return optimized_prompt

    def _all_correct_exception(self, cur_prompt, forward_output, correct_string, helper_data):

        gradient = self.cal_gradient(
            cur_prompt=cur_prompt,
            example_string=correct_string,
            gradient_prompt_template=self.ascend_gradient_prompt_template)

        trajectory_prompts = helper_data['trajectory_prompts']
        trajectory_prompts = self._build_prompt_trajectory_str(trajectory_prompts)

        gradient_descent_output = forward_output
        optimized_prompts = self.optimize(
            cur_prompt=cur_prompt,
            example_string=correct_string,
            gradient=gradient,
            trajectory_prompts=trajectory_prompts,
            steps_per_gradient=self.num_new_prompts,
            optimize_prompt_template=self.ascend_optimize_prompt_template)

        gradient_descent_output['example_string'] = correct_string
        gradient_descent_output['gradient'] = gradient
        gradient_descent_output['optimized_prompts'] = optimized_prompts
        return gradient_descent_output

    def gradient_descent_step(self, cur_prompt, batch, helper_data) -> Dict[str, Any]:

        logger.info(f'cur_prompt: {cur_prompt}')

        forward_output = self.forward(batch=batch, cur_prompt=cur_prompt)
        error_string, correct_string = self._split_error_and_correct_examples(forward_output=forward_output)

        if forward_output['acc'] == 1:
            gradient_descent_output = self._all_correct_exception(
                cur_prompt=cur_prompt,
                forward_output=forward_output,
                correct_string=correct_string,
                helper_data=helper_data)
            return gradient_descent_output

        gradient = self.cal_gradient(
            cur_prompt=cur_prompt,
            example_string=error_string,
            gradient_prompt_template=self.gradient_prompt_template)

        trajectory_prompts = helper_data['trajectory_prompts']
        trajectory_prompts = self._build_prompt_trajectory_str(trajectory_prompts)

        optimized_prompts = self.optimize(
            cur_prompt=cur_prompt,
            example_string=error_string,
            gradient=gradient,
            trajectory_prompts=trajectory_prompts,
            steps_per_gradient=self.num_new_prompts,
            optimize_prompt_template=self.optimize_prompt_template)

        gradient_descent_output = forward_output
        gradient_descent_output['example_string'] = error_string
        gradient_descent_output['gradient'] = gradient
        gradient_descent_output['optimized_prompts'] = optimized_prompts
        return gradient_descent_output

    def __call__(self, batch, cur_prompt, helper_data=None) -> Dict[str, Any]:
        gradient_descent_output = self.gradient_descent_step(cur_prompt=cur_prompt, batch=batch,
                                                             helper_data=helper_data)
        return gradient_descent_output
