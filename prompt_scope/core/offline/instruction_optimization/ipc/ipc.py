import json
import os
import pickle
import asyncio
# import wandb
import random
from pathlib import Path
from typing import List, Any, Literal, Dict
from pydantic import Field, BaseModel
from loguru import logger
import math
from sklearn.metrics import confusion_matrix
from datetime import datetime, date

from prompt_scope.core.offline.instruction_optimization.base_algorithm import PromptOptimizationWithFeedback
from prompt_scope.core.evals.schema import StringEvaluator
from prompt_scope.core.llms.base import BaseLLM
from prompt_scope.core.llms.dashscope_llm import DashscopeLLM

# from prompt_scope.core.models.generation_model import GenerationModel, OpenAIGenerationModel, OpenAIPostModel
class PredictSchema(BaseModel): 
    sample: str = Field(...)
    prediction: str = Field(...)

class SampleSchema(BaseModel): 
    query: str = Field(...)
    answer: str = Field(...)

class IPCOptimization(PromptOptimizationWithFeedback):
    """
    This class implements the Intent-based Prompt Calibration (IPC) algorithm (Intent-based Prompt Calibration: Enhancing prompt optimization with synthetic boundary cases).
    It is designed to refine instructional prompts for language models by iteratively
    generating adversarial samples, evaluating them, updating the prompts based on feedback,
    and repeating the process to enhance prompt effectiveness over multiple iterations.
    """
    # =============LLM Configuration=============
    generation_llm: BaseLLM = Field(default=DashscopeLLM(max_retries=3))
    predictor_llm: BaseLLM = Field(default=DashscopeLLM(max_retries=1))
    analyzer_llm: BaseLLM = Field(default=DashscopeLLM(max_retries=1))
    evaluate_llm: BaseLLM = Field(default=DashscopeLLM(max_retries=1))
    annotate_llm: BaseLLM = Field(default=DashscopeLLM(max_retries=1))

    # =============Path Configuration=============
    prompt_path: str = Field(default=__file__, description="Prompt file path")
    store_path: str = Field(Path(__file__).parent.joinpath("ipc_output"))

    # =============Basic Configuration=============
    task_type: Literal["classification", "generation"] = Field(...)
    label_schema: List[str] = Field(default=[])
    task_description: str = Field(...)
    cur_step: int = Field(default=0)  # Tracks the current step in the iterative process
    history: List = []

    # =============Experiment Configuration=============
    samples_per_step: int = Field(default=10, description="samples generated for each step")
    batch_size: int = Field(default=10, description="number of samples generated in single LLM call")
    max_samples: int = Field(default=50, description="maximum samples in storage")
    warmup: int = Field(default=4, description="warmup epochs: patience and prompt update may be disabled")
    history_length: int = Field(default=4, description="History length to look back")
    num_errors_per_label: int = Field(default=5, alias='num_errors',
                                      description="Number of errors to extract from the evaluator for each label")
    num_extra_sample: int = Field(default=5, description="Extra samples to be included during step generation")
    patience: int = Field(default=10, description="Patience for early stopping")
    min_delta: float = Field(default=0.1, description="Minimum improvement to clear patience")
    max_usage: int = Field(default=10000)
    samples: List[str] = Field(default=[])
    annotations: List[str] = Field(default=[])
    predictions: List[str] = Field(default=[])

        
    def _before_run(self):
        pass
    
    def _after_run(self):
        output_path = create_dated_directory(self.store_path)
        output_path = Path(output_path)
        self.save_state(output_path)
        return self.extract_best_prompt()

    def _step(self, *, i_step: int) -> bool:
        """
        Executes the main iteration step of the IPC optimization process. This involves generating or processing samples,
        evaluating their performance, updating the current prompt, and checking the stop criteria.

        Returns:
          bool: Returns `True` if the stop criteria are met; otherwise, `False`.
        """
        logger.info(f'Starting step {i_step}')

        if self.task_type == 'generation':
            prompt_type = 'adv_sample_generation'
        else:
            prompt_type = 'adv_sample_classification'

        if not (self.data_path or self.samples):
            logger.info('Dataset is empty generating initial samples')
            prompt_input = {
                'task_description': self.task_description, 
                'instruction': self.instruction,
                'batch_size': self.batch_size
                }
            generate_prompt = getattr(self.prompt_handler, prompt_type) \
                .format_map(prompt_input)
            # new_samples = asyncio.run(self._agenerate(prompt=generate_prompt))
            # in case you want to use batch:
            new_samples = self._generate(prompt=generate_prompt)
            self.samples.extend(new_samples)
        else:
            logger.info('Generating Adversarials')
            new_samples = self._step_generate()
            if not new_samples:
                return True
            self.samples.extend(new_samples)
        
        if self.task_type == 'generation':
            logger.info('Calculating Score and Error Analysis')
        elif self.task_type == 'classification':
            logger.info('Annotating')
            new_annotations = self._predict(
                samples=new_samples, 
                llm=self.annotate_llm)
            self.annotations.extend(new_annotations)

            logger.info('Running predictor')
            new_predictions = self._predict(
                samples=new_samples, 
                llm=self.predictor_llm)
            self.predictions.extend(new_predictions)
            
            logger.info('Calculating Score and Error Analysis')
            history = self._evaluate_and_analyze(
                input=new_samples,
                reference=new_annotations, 
                prediction=new_predictions, 
                evaluator=self.evaluator, 
                label_schema=self.label_schema)
        
        logger.info('Updating Prompt')
        self._update_prompt(history)
        if self.stop_criteria():
            logger.info('Stop criteria reached')
            return True
        self.cur_step = i_step
        return False

    def _predict(self, *, samples: List[str], llm: BaseLLM) -> List[PredictSchema]:
        """
        predicts a list of text samples using a Large Language Model (LLM) (Argilla annotation to be implemented),
        following the instructions and batch size configurations.

        Args:
            samples (list[str]): A list of strings, where each string is a sample to be predicted.
            llm (BaseLLM): the predictor LLM

        Returns:
        list[PredictSchema]: Each schema contains fields 'sample' and 'prediction'.
        """
        class PredictsSchema(BaseModel):
            predictions: List[PredictSchema]= Field(..., min_length=self.batch_size, max_length=self.batch_size)
        
        samples_batches = [samples[i:i + self.batch_size] for i in
                           range(0, len(samples), self.batch_size)]
        results = []
        for sample_batch in samples_batches:
            sample_str = "|".join(sample_batch)
            prompt_input = {'samples': sample_str, 'instruction': self.instruction,
                            'batch_size': len(sample_batch), 'label_schema': self.label_schema}
            predict_prompt = self.prompt_handler.predict_batch.format_map(prompt_input)
            results.extend(llm.structured_output(
                messages=predict_prompt,
                schema=PredictsSchema).predictions)
        logger.info(results)
        return results
    
    def extract_best_prompt(self) -> Dict[str, Any]:
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
            self.history[min(self.warmup - 1, len(self.history) - 1):],
            key=lambda x: x['score'],
            reverse=False)

        # Return the prompt and score of the entry with the best score
        return {'prompt': sorted_history[-1]['prompt'], 'score': sorted_history[-1]['score']}

    def _update_prompt(self, history):
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
        step_num = len(history)
        if (step_num < self.warmup) or (step_num % 3) > 0:
            last_history = history[-self.history_length:]
        else:
            sorted_history = sorted(history[self.warmup - 1:], key=lambda x: x['score'],
                                    reverse=False)
            last_history = sorted_history[-self.history_length:]

        history_prompt = '\n'.join([f"####\n##Prompt Score: {history['score']:.2f}\n##Prompt:\n{history['instruction']}\n#################\n"for history in last_history])
        prompt_input = {"history": history_prompt, "task_description": self.task_description,
                        'error_analysis': last_history[-1]['analysis'], "labels": self.label_schema}
        if self.task_type == 'generation':
            generate_prompt = self.prompt_handler.update_prompt_generation \
                .format_map(prompt_input)
        elif self.task_type == 'classification':
            generate_prompt = self.prompt_handler.update_prompt_classification \
                .format_map(prompt_input)
        prompt_suggestion = self.generation_llm.chat(messages=generate_prompt).message.content
        logger.info(f'Get new prompt:\n{prompt_suggestion}')
        self.instruction = prompt_suggestion

    def _evaluate_and_analyze(self, 
                              *, 
                              input: List[str]=[],
                              reference: List[PredictSchema]=[], 
                              prediction: List[PredictSchema],
                              evaluator: StringEvaluator,
                              label_schema: Any) -> List[Dict[str, Any]]:
        
        if self.task_type == 'generation':
            return self._evaluate_and_analyze_generation(
                input=input, reference=reference, prediction=prediction, 
                evaluator=evaluator, label_schema=label_schema)
        elif self.task_type == 'classification':
            return self._evaluate_and_analyze_classfication(
                input=input, reference=reference, prediction=prediction, 
                evaluator=evaluator, label_schema=label_schema)
        
    def _evaluate_and_analyze_classfication(self, 
                              *, 
                              input: List[str]=[],
                              reference: List[PredictSchema]=[], 
                              prediction: List[PredictSchema],
                              evaluator: StringEvaluator,
                              label_schema: Any) -> List[Dict[str, Any]]:

        scores = list(map(lambda p, r: 
                                evaluator.evaluate_strings(prediction=p, reference=r), 
                                [x.prediction for x in prediction], [x.prediction for x in reference]))
        correct_indices = [i for i, d in enumerate(scores) if d['score'] == 1]
        accuracy = len(correct_indices) / len(scores)
        error_indices = [[i for i, d in enumerate(scores) if d['score'] == 0 and reference[i] == label] for label in label_schema]
        correct_sample = [prediction[i] for i in correct_indices]
        error_sample = [[prediction[i] for i in error_indices_label] for error_indices_label in error_indices]
        conf_matrix = confusion_matrix(y_true=[x.prediction for x in reference], y_pred=[x.prediction for x in prediction], labels=label_schema)

        # error_to_str
        error_str = ''
        for error_indices_label in error_indices:
            for idx in error_indices_label[-self.num_errors_per_label:]:
                error_str += f"Sample: {input[idx]}\nPrediction: {prediction[idx].prediction}, GT: {reference[idx].prediction}\n#\n"

        conf_text = f"Confusion matrix columns:{self.label_schema} the matrix data:"
        for i, row in enumerate(conf_matrix):
            conf_text += f"\n{self.label_schema[i]}: {row}"
        prompt_input = {"task_description": self.task_description, 
                        "instruction": self.instruction,
                        "score": accuracy,
                        "confusion_matrix": conf_text,
                        "failure_cases": error_str}
        analyze_prompt = self.prompt_handler.error_analysis_classification.format_map(prompt_input)
        analysis = self.analyzer_llm.chat(messages=analyze_prompt).message.content
        prompt_input['analysis'] = analysis
        prompt_input['error_str'] = error_str
        prompt_input['errors'] = error_sample
        self.history.append(prompt_input)
        return self.history
    
    def _generate(self, prompt: str):
        """
        Generates a list of samples based on the given prompt using the generation_llm.
        Args:
            prompt (str): The initial instructional prompt to generate samples from.
            
        Returns:
            List[str]: A list of generated samples, each a possible variation or response to the input prompt.
        """
        class SamplesSchema(BaseModel):
            samples: List[SampleSchema]= Field(..., min_length=self.batch_size, max_length=self.batch_size)
        
        num_batches = math.ceil(self.samples_per_step / self.batch_size)
        new_samples = []
        for _ in range(num_batches):
            new_samples += [x.query for x in self.generation_llm.structured_output(
            messages=prompt,
            schema=SamplesSchema).samples]
        logger.info(new_samples)
        return new_samples
    
    async def _agenerate(self, prompt: str) -> List[SampleSchema]:
        """
        Generates a list of samples based on the given prompt asynchronously.
        Args:
            prompt (str): The initial instructional prompt to generate samples from.
            
        Returns:
            List[str]: A list of generated samples, each a possible variation or response to the input prompt.
        """
        prompts = [prompt] * self.samples_per_step
        samples_list = await self.generation_llm.astructured_output(
            list_of_messages=prompts,
            schema=SampleSchema)
        logger.info(samples_list)
        return samples_list

    def _step_generate(self):
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
        if len(self.samples) < self.max_samples:
            step_num = len(self.history)
            if (step_num < self.warmup) or (step_num % 3) > 0:
                last_history = self.history[-self.history_length:]
            else:
                sorted_history = sorted(self.history[self.warmup - 1:], key=lambda x: x['score'],
                                        reverse=False)
                last_history = sorted_history[-self.history_length:]

            indices = random.choices(range(len(self.samples)), k=self.num_extra_sample)
            extra_samples_text = '##\n'
            for sample in [self.samples[i] for i in indices]:
                extra_samples_text += f"Sample:\n {sample}\n#\n"
            prompt_input = {'task_description': self.task_description, 'instruction': self.instruction,
                            'batch_size': self.batch_size, 'extra_samples': extra_samples_text}
            if sum([len(t['errors']) for t in last_history]) > 0:
                history_samples = "\n".join([f"####\n##Prompt:\n{sample['instruction']}\n{sample['error_str']}####\n"
                                             for sample in last_history])
                prompt_input['history'] = history_samples
            else:
                prompt_input['history'] = 'No previous errors information'
            generate_prompt = self.prompt_handler.step_adv_sample_classification.format_map(prompt_input)
            new_samples = self._generate(prompt=generate_prompt)
            self.samples.extend(new_samples)
            logger.info(self.samples)
            return new_samples
        else:
            return []

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
        if len(self.history) <= self.warmup:
            self.patient = 0
            return False

        if len(self.history) == 1:
            max_score = 0
        else:
            max_score = sorted(self.history[:-1], lambda x: x['score'])[-1]['score']

        if max_score - self.history[-1]['score'] > -self.min_delta:
            self.patient += 1
        else:
            self.patient = 0

        if self.patient > self.patience:
            return True
        return False

    def save_state(self, output_path):
        """
        Saves the current state of the iterative process, including evaluation history,
        the current step, prompts, task description, and patient data (if applicable),
        to a pickle file. This function is called to persist the progress made during
        the optimization of instructional prompts.
        """
        logger.info('Save state')

        state = {'history': self.history, 'step': self.cur_step,
                 'prompt': self.instruction, 'task_description': self.task_description,
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

        try:
            mod_prompt = self.generation_llm.call(prompt=prompt_mod_prompt).message.content
            mod_description = self.generation_llm.call(prompt=description_mod_prompt).message.content
        except Exception:
            mod_prompt = self.generation_llm.call(prompt=prompt_mod_prompt).output.text
            mod_description = self.generation_llm.call(prompt=description_mod_prompt).output.text
        logger.info(f"Task description modified for ranking to: \n{mod_description}")

        logger.info(f"Initial prompt modified for ranking to: \n{mod_prompt}")

        self.task_config.instruction = mod_prompt
        self.task_config.task_description = mod_description


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