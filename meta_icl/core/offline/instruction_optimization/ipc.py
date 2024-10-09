import json
import os
import pickle
# import wandb
import random
from pathlib import Path
from typing import List

from loguru import logger

from meta_icl import CONFIG_REGISTRY
from meta_icl.algorithm.base_algorithm import PromptOptimizationWithFeedback
from meta_icl.core.evaluation.evaluator import Eval
from meta_icl.core.models.generation_model import GenerationModel, OpenAIGenerationModel, OpenAIPostModel
from meta_icl.core.offline.demonstration_augmentation.ipc_aug import IPCGeneration


class IPCOptimization(PromptOptimizationWithFeedback):
    """
    This class implements the Intent-based Prompt Calibration (IPC) algorithm (Intent-based Prompt Calibration: Enhancing prompt optimization with synthetic boundary cases).
    It is designed to refine instructional prompts for language models by iteratively
    generating adversarial samples, evaluating them, updating the prompts based on feedback,
    and repeating the process to enhance prompt effectiveness over multiple iterations.
    """
    FILE_PATH: str = __file__

    def __init__(self, language="cn", **kwargs):
        """
        Initializes the IPC Optimization instance with necessary configurations, model setup,
        and initializes key attributes used throughout the iterative prompt refinement process.

        Args:
            language (str): The language setting for the optimization process, defaulting to "cn".
            **kwargs: Additional keyword arguments passed to the superclass initializer.

        The method sets up the initial configuration, initializes the model tailored for the task,
        and sets default values or placeholders for properties crucial to the algorithm's workflow,
        such as patience level, sample texts, current step counter, initial prompt, and an evaluation object.
        """
        super().__init__(language=language, **kwargs)

        self.init_config()  # ⭐ Initialize the configuration for optimization
        self.init_model()  # ⭐ Set up the model used in the optimization process
        self.patient: int = 0  # Patience counter for optimization steps
        self.samples: List[str] = None  # Placeholder for generated sample texts
        self.cur_step: int = 0  # Tracks the current step in the iterative process
        self.cur_prompt: str = self.task_config.instruction  # Initial prompt instruction
        # todo: by jm, eval module is only used in ipc optimization?
        self.eval = Eval(FILE_PATH=self.FILE_PATH)  # Instantiate evaluation module with file path

    def init_model(self):
        """
        Initializes the language models required for the IPC_Optimization process.
        This includes setting up the generation, predictor, and annotator models
        with configurations specified in `self.model_config`.

        The method uses the `GenerationModel` class to instantiate these models,
        passing respective configuration parameters for each model's unique role.
        """
        print(self.model_config)
        generation_module_name = self.model_config.generation.get('module_name')
        if generation_module_name == 'dashscope_generation':
            self.generation_llm = GenerationModel(**self.model_config.generation)
        elif generation_module_name == 'openai_generation':
            self.generation_llm = OpenAIGenerationModel(**self.model_config.generation)
        elif generation_module_name == 'openai_post':
            self.generation_llm = OpenAIPostModel(**self.model_config.generation)

        predictor_module_name = self.model_config.predictor.get('module_name')
        if predictor_module_name == 'dashscope_generation':
            self.predictor_llm = GenerationModel(**self.model_config.predictor)
        elif predictor_module_name == 'openai_generation':
            self.predictor_llm = OpenAIGenerationModel(**self.model_config.predictor)
        elif predictor_module_name == 'openai_post':
            self.predictor_llm = OpenAIPostModel(**self.model_config.predictor)

        annotator_module_name = self.model_config.annotator.get('module_name')
        if annotator_module_name == 'dashscope_generation':
            self.annotator_llm = GenerationModel(**self.model_config.annotator)
        elif annotator_module_name == 'openai_generation':
            self.annotator_llm = OpenAIGenerationModel(**self.model_config.annotator)
        elif annotator_module_name == 'openai_post':
            self.annotator_llm = OpenAIPostModel(**self.model_config.annotator)

    def init_config(self):
        """
        Initializes the configuration for the IPC_Optimization process by fetching necessary configurations from the registry.

        This method sets up configurations for tasks, models, rankers, and evaluation, ensuring all components have their
        respective settings ready before the optimization pipeline begins. If the class has an 'eval' attribute, its configuration
        is also initialized.
        """
        self.task_config = CONFIG_REGISTRY.module_dict['task_config']
        self.model_config = CONFIG_REGISTRY.module_dict['model_config']
        self.ranker_config = CONFIG_REGISTRY.module_dict.get('ranker_config', None)
        self.eval_config = CONFIG_REGISTRY.module_dict.get('eval_config', None)
        if hasattr(self, 'eval'):
            self.eval.init_config()

    def run(self, **kwargs):
        """
        Executes the main optimization loop of the IPC Optimization process for a specified number of steps.

        Args:
            **kwargs: Additional keyword arguments that can modify the behavior of the optimization, including:
                - mode (str, optional): If set to 'ranking', adjusts input handling for ranking tasks.
                - ranking_prompt (str, optional): Custom evaluation instruction when ranking prompts.

        The method iterates for 'num_steps', potentially modifying inputs based on 'mode', updating the evaluation
        instruction with 'ranking_prompt', and checks for stop criteria after each step. Finally, it returns the best
        refined instructional prompt post-optimization.
        """
        # Run the optimization pipeline for num_steps
        if kwargs.get('mode', '') == 'ranking':
            self.modify_input_for_ranker()
        if 'ranking_prompt' in kwargs:
            self.eval.eval_instruction = kwargs['ranking_prompt']
        for _ in range(self.task_config.num_steps):
            stop_criteria = self.step(**kwargs)
            if stop_criteria:
                break
        final_result = self.extract_best_prompt()
        return final_result

    def step(self, **kwargs):
        """
        Executes the main iteration step of the IPC optimization process. This involves generating or processing samples,
        evaluating their performance, updating the current prompt, and checking the stop criteria.

        Args:
            **kwargs: Additional keyword arguments including:
                - mode (str, optional): Specifies the operation mode, either 'generation' or 'classification'. Defaults to 'classification'.

        Returns:
          bool: Returns `True` if the stop criteria are met; otherwise, `False`.
        """
        logger.info(f'Starting step {self.cur_step}')
        if kwargs.get('mode', '') == 'generation':
            prompt_type = 'adv_sample_generation'
        else:
            prompt_type = 'adv_sample_classification'

        mode = kwargs.get('mode', 'classification')

        if not hasattr(kwargs, 'data'):
            if not self.samples:
                logger.info('Dataset is empty generating initial samples')
                prompt_input = {'task_description': self.task_config.task_description, 'instruction': self.cur_prompt,
                                'batch_size': self.task_config.batch_size}
                generate_prompt = getattr(self.prompt_handler, prompt_type).format_map(prompt_input)
                self.samples = self.generate(prompt=generate_prompt)
            else:
                logger.info('Generating Adversarials')
                self.step_generate()

        samples = [sample.split('|') for sample in self.samples]
        eval_kwargs = {}
        eval_kwargs['prompt'] = self.cur_prompt

        if kwargs.get('mode', '') == 'generation':
            logger.info('Calculating Score and Error Analysis')
            self.eval.init_config()
            eval_kwargs['score'], eval_kwargs['errors'] = self.eval.eval_with_llm(
                samples=[sample[-1] for sample in samples], prompt_handler=self.prompt_handler)
        else:
            # todo: by zy, i have checked the prompt used in annotator and predictor, which are almost the same.
            #  If we set the same llm for annotator and predictor, I am little confused by this evaluation.
            logger.info('Running annotator')
            annotations = self.annotate([sample[1].strip() if len(sample) > 1 else '' for sample in samples])
            logger.info('Running predictor')
            predictions = self.predict([sample[1].strip() if len(sample) > 1 else '' for sample in samples])
            logger.info('Calculating Score and Error Analysis')
            eval_kwargs['score'], eval_kwargs['corrects'], eval_kwargs['errors'], eval_kwargs[
                'conf_matrix'] = self.eval.eval_accuracy(annotations, predictions, self.task_config.label_schema)
        self.eval.error_analysis(prompt_handler=self.prompt_handler, **eval_kwargs)
        logger.info('Updating Prompt')
        self.update_cur_prompt(mode)
        if self.stop_criteria():
            logger.info('Stop criteria reached')
            return True
        self.save_state(mode)
        self.cur_step += 1
        return False

    def annotate(self, samples: list[str]):
        """
        Annotates a list of text samples using a Large Language Model (LLM) (Argilla to be implemented),
        following the instructions and batch size configurations set in `task_config`.

        The method divides the input samples into batches, constructs an annotation prompt
        for each batch, sends these prompts to the annotator (LLM), processes the responses
        to extract annotations, and finally aggregates these into a list of dictionaries
        containing IDs, questions, and corresponding annotations.

        Args:
            samples (list[str]): A list of strings, where each string is a sample to be annotated.

        Returns:
        list[dict]: A list of dictionaries. Each dictionary contains keys 'ID', '问题' (Question),
                    and '标注' (Annotation), representing the annotated data.
        """
        samples_batches = [samples[i:i + self.task_config.batch_size] for i in
                           range(0, len(samples), self.task_config.batch_size)]
        batch, annotations = 0, []
        for sample_batch in samples_batches:
            sample_str = "|".join(sample_batch)
            annotate_prompt = self.prompt_handler.annotate.format(samples=sample_str,
                                                                  instruction=self.task_config.instruction,
                                                                  batch_size=self.task_config.batch_size)
            # print('#############\n', annotate_prompt, '################\n')
            response = self.annotator_llm.call(prompt=annotate_prompt)
            try:
                response_list = [item for item in response.message.content.split("||") if item]
            except:
                response_list = [item for item in response.output.text.split("||") if item]
            # print('#############\n', response_list, '################\n')
            annotations.extend([{"ID": f"{batch}_{lines[0].strip()}", "问题": lines[1], "标注": lines[-1]} for lines in
                                (sample.split('|') for sample in response_list if sample)])
            batch += 1
        logger.info(annotations)
        return annotations

    def predict(self, samples: list[str]):
        """
        Generates predictions for a list of samples by dividing them into batches,
        processing each batch with the annotator, and formatting the responses.

        Args:
            samples (list[str]): A list of strings, where each string represents a sample.

        Returns:
            list[dict]: A list of dictionaries, each containing the ID, question, and prediction
                        for a processed sample.
        """
        # Divide samples into batches based on the configured batch size
        samples_batches = [samples[i:i + self.task_config.batch_size] for i in
                           range(0, len(samples), self.task_config.batch_size)]
        batch, predictions = 0, []
        for sample_batch in samples_batches:
            sample_str = "|".join(sample_batch)
            prediction_prompt = self.prompt_handler.predict.format(samples=sample_str,
                                                                   instruction=self.task_config.instruction,
                                                                   batch_size=self.task_config.batch_size)
            # print('#############\n', prediction_prompt, '################\n')
            response = self.predictor_llm.call(prompt=prediction_prompt)
            try:
                response_list = [item for item in response.message.content.split("||") if item]
            except:
                response_list = [item for item in response.output.text.split("||") if item]
            # print('#############\n', response_list, '################\n')
            predictions.extend([{"ID": f"{batch}_{lines[0].strip()}", "问题": lines[1], "预测": lines[-1]} for lines in
                                (sample.split('|') for sample in response_list if sample)])
            batch += 1
        logger.info(predictions)
        return predictions

    def extract_best_prompt(self):
        """
        Extracts the best prompt from the evaluation history based on the score.

        The function sorts the recent history of evaluations, excluding any warmup iterations,
        and identifies the prompt with the lowest score (assuming lower scores are better).
        It returns a dictionary containing the best prompt and its associated score.

        Returns:
            dict: A dictionary with keys 'prompt' and 'score', where 'prompt' is the best
                  prompt found and 'score' is its corresponding evaluation metric value.
        """
        # Sort the evaluation history based on the score, considering only entries after the warmup period
        sorted_history = sorted(
            self.eval.history[min(self.task_config.warmup - 1, len(self.eval.history) - 1):],
            key=lambda x: x['score'],
            reverse=False)

        # Return the prompt and score of the entry with the best score
        return {'prompt': sorted_history[-1]['prompt'], 'score': sorted_history[-1]['score']}

    def update_cur_prompt(self, mode):
        """
        Updates the current prompt by generating a new prompt suggestion based on historical data and task description.
        It also estimates the score of the previous prompt and prepares a set of challenging samples for the updated prompt.

        Args:
            mode (str): Specifies the mode of operation for prompt generation, e.g., 'generation' or other modes defined.

        Steps:
        1. Determines the subset of historical samples to use for prompt generation, considering warmup steps and frequency.
        2. Constructs a 'history_prompt' string from the selected historical samples.
        3. Forms the input for the prompt generation including history, task description, and error analysis.
        4. Optionally includes label schema in the prompt input if available in the task configuration.
        5. Chooses the appropriate prompt template based on the 'mode'.
        6. Generates a new prompt suggestion using a language model.
        7. Logs the new prompt suggestion and the previous prompt's mean score.
        8. Assigns the generated prompt as the current prompt for further use.

        Note:
            - The function handles exceptions during the LLM call to ensure prompt retrieval.
            - Warmup steps might skip prompt updates initially to stabilize the process.
        """
        step_num = len(self.eval.history)
        if (step_num < self.task_config.warmup) or (step_num % 3) > 0:
            self.last_history = self.eval.history[-self.task_config.history_length:]
        else:
            sorted_history = sorted(self.eval.history[self.task_config.warmup - 1:], key=lambda x: x['score'],
                                    reverse=False)
            self.last_history = sorted_history[-self.task_config.history_length:]
        history_prompt = '\n'.join([self.eval.sample_to_text(sample) for sample in self.last_history])
        prompt_input = {"history": history_prompt, "task_description": self.task_config.task_description,
                        'error_analysis': self.last_history[-1]['analysis']}
        if hasattr(self.task_config, 'label_schema'):
            prompt_input["labels"] = json.dumps(self.task_config.label_schema)
        if mode == 'generation':
            generate_prompt = self.prompt_handler.prompt_generation_generation.format_map(prompt_input)
        else:
            generate_prompt = self.prompt_handler.prompt_generation.format_map(prompt_input)
        try:
            prompt_suggestion = self.generation_llm.call(prompt=generate_prompt).message.content
        except:
            prompt_suggestion = self.generation_llm.call(prompt=generate_prompt).output.text
        logger.info(prompt_suggestion)
        logger.info(f'Previous prompt score:\n{self.eval.mean_score}\n#########\n')
        logger.info(f'Get new prompt:\n{prompt_suggestion}')
        self.cur_prompt = prompt_suggestion

    def generate(self, prompt: str):
        """
        Generates a list of samples based on the given prompt using the configured language model.
        The function first creates a batch of inputs from the prompt, then distributes the generation task
        across multiple workers. It handles both cases where the output is in the message content or the text property.
        After collecting all generated samples, it returns a flattened list of non-empty, stripped samples.

        Args:
            prompt (str): The initial instructional prompt to generate samples from.

        Returns:
            List[str]: A list of generated samples, each a possible variation or response to the input prompt.
        """
        batch_input = prompt
        batch_inputs = IPCGeneration.generate_samples_batch(batch_input, self.task_config.samples_per_step,
                                                            self.task_config.batch_size)
        samples_batches = IPCGeneration.batch_call(batch_inputs, self.task_config.workers, self.generation_llm)
        try:
            samples_lists = [samples_batch.message.content.split("||") for samples_batch in samples_batches]
        except:
            samples_lists = [samples_batch.output.text.split("||") for samples_batch in samples_batches]
        samples_list = [item.strip() for sample_list in samples_lists for item in sample_list if item]
        logger.info(samples_list)
        return samples_list

    def step_generate(self):
        """
        Generates new samples based on updated prompts, incorporating extra samples and historical data when available.

        This step is part of an iterative process to refine instructional prompts for language models. It assembles a new
        prompt with the current task description, the current prompt being evaluated, a shuffled subset of existing samples,
        and optionally, historical samples that have been previously evaluated as error-free. The constructed prompt is then
        used to generate additional samples via an external generation function.

        Steps Involved:
        1. Checks if the number of existing samples is less than the maximum allowed.
        2. Prepares a dictionary (`prompt_input`) containing necessary information for generating new prompts:
           - Task description.
           - Current prompt in evaluation.
           - A batch size for generation.
           - A text summary of extra samples for context.
           - Historical samples, if all past samples have no error labels; otherwise, indicates no prior error info.
        3. Constructs the new prompt using the prepared information and a predefined format.
        4. Generates new samples using the constructed prompt.
        5. Augments the existing samples list with the newly generated ones.
        6. Logs the updated samples list for monitoring.

        Note: The effectiveness of this method relies on the external functions for sample handling, prompt formatting,
        and the actual generation process, which are not shown here.
        """
        if len(self.samples) < self.task_config.max_samples:
            prompt_input = {'task_description': self.task_config.task_description, 'prompt': self.cur_prompt,
                            'batch_size': self.task_config.batch_size}
            random.shuffle(self.samples)
            txt_res = '##\n'
            for sample in self.samples[:self.task_config.num_extra_sample]:
                txt_res += f"Sample:\n {sample}\n#\n"
            prompt_input['extra_samples'] = txt_res
            if all([all(len(error_label) == 0 for error_label in t['errors']) for t in self.last_history]):
                history_samples = '\n'.join([self.eval.sample_to_text(sample,
                                                                      num_errors_per_label=self.task_config.num_errors_per_label,
                                                                      is_score=False) for sample in self.last_history])
                prompt_input['history'] = history_samples
            else:
                prompt_input['history'] = 'No previous errors information'
            generate_prompt = self.prompt_handler.step_adv_sample_classification.format_map(prompt_input)
            new_samples = self.generate(prompt=generate_prompt)
            self.samples.extend(new_samples)
            logger.info(self.samples)

    def stop_criteria(self):
        """
        Determines if the optimization process should stop based on predefined criteria.
        The algorithm stops when:
        1. The improvement in score has not exceeded a minimum threshold over a certain number of consecutive steps ('patient' steps),
           after an initial warmup period.
        2. Optionally, checks if resource usage exceeds a set limit, but this check is currently disabled.

        Returns:
            bool: True if the stop criteria are met, indicating the optimization should end; otherwise, False.
        """
        # if 0 < self.config.stop_criteria.max_usage < self.calc_usage():
        #     return True
        if len(self.eval.history) <= self.task_config.warmup:
            self.patient = 0
            return False

        if len(self.eval.history) == 1:
            max_score = 0
        else:
            max_score = self.eval.get_max_score(self.task_config.warmup)

        if max_score - self.eval.history[-1]['score'] > -self.task_config.min_delta:
            self.patient += 1
        else:
            self.patient = 0

        if self.patient > self.task_config.patience:
            return True
        return False

    def save_state(self, mode):
        """
        Saves the current state of the iterative process, including evaluation history,
        the current step, prompts, task description, and patient data (if applicable),
        to a pickle file. This function is called to persist the progress made during
        the optimization of instructional prompts.

        Args:
            mode (str): The mode indicating the specific state or configuration to save.
                         Determines the subdirectory within the output path where the state will be saved.

        Returns:
        None

        Note:
            - The function checks if the 'output_path' attribute exists in 'task_config' and is not None before proceeding.
            - If the specified directory does not exist, it will be created.
            - The state is pickled into a file named 'history.pkl' within the respective mode's directory.
        """
        if not mode:
            return
        if not hasattr(self.task_config, 'output_path') or self.task_config.output_path is None:
            return
        logger.info('Save state')
        output_path = os.path.join(self.task_config.output_path, mode)
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        output_path = Path(output_path)

        state = {'history': self.eval.history, 'step': self.cur_step,
                 'prompt': self.cur_prompt, 'task_description': self.task_config.task_description,
                 'patient': self.patient}
        pickle.dump(state, open(output_path / 'history.pkl', 'wb'))

    def load_state(self, path: str):
        """
        Loads the pre-trained state of the IPC_Optimization instance from a specified directory.

        This method reads the 'history.pkl' file located within the given path, which contains essential
        information about the past optimization process, including the history of evaluations, the current
        batch identifier, the active prompt being refined, a description of the task, and details about
        the patience configuration for the algorithm's learning strategy.

        Args:
            path (str): The file system path to the directory where the 'history.pkl' file is stored.

        Note:
            The state includes:
            - 'history': A record of evaluation metrics across iterations.
            - 'batch_id': An identifier indicating the batch of prompts being processed.
            - 'prompt': The current instructional prompt being optimized.
            - 'task_description': A description outlining the objective or context of the optimization task.
            - 'patient': Parameters governing the patience in the optimization, e.g., when to adjust prompts.
        """
        path = Path(path)
        if (path / 'history.pkl').is_file():
            state = pickle.load(open(path / 'history.pkl', 'rb'))
            self.eval.history = state['history']
            self.batch_id = state['batch_id']
            self.cur_prompt = state['prompt']
            self.task_description = state['task_description']
            self.patient = state['patient']

    def modify_input_for_ranker(self):
        """
        Modifies the input prompt and task description for the ranker by utilizing LLM-generated modifications.

        This method takes the initial instruction and task description from the task configuration, formats them
        into prompts for an LLM to modify, calls the LLM to generate these modifications, and then updates the
        task configuration with these new, refined inputs. It handles both successful structured responses
        (with `.message.content`) and fallback scenarios where the response structure might differ
        (using `.output.text`).

        Raises:
            Any exceptions raised during the LLM call attempts are caught internally and handled by falling back
            to a different attribute extraction method.

        Logs:
            Information about the modified task description and initial prompt after the modification process.
        """
        prompt_input = {'label_schema': self.task_config.label_schema,
                        'prompt': self.task_config.instruction}
        prompt_mod_prompt = self.prompt_handler.ipc_ranker_prompt_mod.format_map(prompt_input)
        prompt_input = {'task_description': self.task_config.task_description}
        description_mod_prompt = self.prompt_handler.ipc_ranker_description_mod.format_map(prompt_input)

        # todo: by zy, unify the llm call interface.
        try:
            mod_prompt = self.generation_llm.call(prompt=prompt_mod_prompt).message.content
            mod_description = self.generation_llm.call(prompt=description_mod_prompt).message.content
        except:
            mod_prompt = self.generation_llm.call(prompt=prompt_mod_prompt).output.text
            mod_description = self.generation_llm.call(prompt=description_mod_prompt).output.text
        logger.info(f"Task description modified for ranking to: \n{mod_description}")

        logger.info(f"Initial prompt modified for ranking to: \n{mod_prompt}")

        self.task_config.instruction = mod_prompt
        self.task_config.task_description = mod_description
