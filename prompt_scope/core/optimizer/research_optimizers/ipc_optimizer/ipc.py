import random
import os
from pydantic import Field, BaseModel
from typing import Literal, List, Dict, Tuple, Any
from loguru import logger
from pathlib import Path
from sklearn.metrics import confusion_matrix
from collections import Counter
from operator import attrgetter
from itertools import groupby

from prompt_scope.core.optimizer.research_optimizers.base_optimizer import (
    OptimizationConfig,
    PromptOptimizationWithFeedback,
)
from prompt_scope.core.llms.base import BaseLLM
from prompt_scope.core.llms.dashscope_llm import DashscopeLLM, DashScopeLlmName


class IPCConfig(OptimizationConfig):
    """Configuration for IPC optimization"""

    # LLM Configuration
    generation_llm: BaseLLM = Field(default=DashscopeLLM(max_retries=3, model=DashScopeLlmName.QWEN2_72B_INST))
    predictor_llm: BaseLLM = Field(default=DashscopeLLM(max_retries=1))
    analyzer_llm: BaseLLM = Field(default=DashscopeLLM(max_retries=1, model=DashScopeLlmName.QWEN2_72B_INST))
    annotate_llm: BaseLLM = Field(default=DashscopeLLM(max_retries=1, model=DashScopeLlmName.QWEN2_72B_INST))

    # Task Configuration
    task_type: Literal["classification", "generation"] = Field(...)
    label_schema: List[str] = Field(default=[])
    task_description: str = Field(...)
    samples: List[str] = Field(default=[])
    store_path: Path = Field(
        default_factory=lambda: Path(os.path.join(Path(__file__).parent, "output")),
        description="Path to the prompt template file",
    )
    prompt_path: Path = Field(
        default_factory=lambda: Path(__file__),
        description="Path to the prompt template file",
    )
    # Optimization Parameters
    samples_per_step: int = Field(default=10)
    max_samples: int = Field(default=50)
    warmup: int = Field(default=4)
    history_length: int = Field(default=4)
    num_errors_per_label: int = Field(default=5)
    num_extra_sample: int = Field(default=5)
    min_delta: float = Field(default=0.1)


class ErrorCase(BaseModel):
    input: str
    prediction: str
    reference: str


class ErrorAnalysis(BaseModel):
    summary: str
    error_cases: Dict[str, List[ErrorCase]]
    error_distribution: Dict[str, int]


class StepResult(BaseModel):
    score: float
    error_analysis: ErrorAnalysis
    instruction: str


class IPCOptimization(PromptOptimizationWithFeedback):
    """Implementation of Intent-based Prompt Calibration algorithm"""

    def __init__(self, config: IPCConfig):
        super().__init__(config)
        self.config: IPCConfig = config
        self.samples: List[str] = config.samples
        self.annotations: List[str] = []
        self.predictions: List[str] = []
        self.history: List[StepResult] = []
        self.patient = 0

    def _before_run(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Initialize IPC optimization process"""
        data_kwargs = {
            "evaluator": self.evaluator,
            "label_schema": self.config.label_schema,
        }

        result_kwargs = {
            "samples": self.samples,
            "annotations": self.annotations,
            "predictions": self.predictions,
            "history": self.history,
        }

        return data_kwargs, result_kwargs

    def _step(self, *, i_step: int, **kwargs) -> bool:
        """Execute single IPC optimization step"""
        try:
            # Generate or process samples
            if not self.samples:
                new_samples = self._handle_empty_samples()
            else:
                new_samples = self._generate_adv_samples()

            if not new_samples:
                return True

            self.samples.extend(new_samples)

            # Handle task-specific processing
            if self.config.task_type == "generation":
                self._handle_generation_task(new_samples, **kwargs)
            elif self.config.task_type == "classification":
                self._handle_classification_task(new_samples, **kwargs)

            # Update prompt based on results
            self._update_prompt(self.history)

            # Check stop criteria
            return self.stop_criteria()

        except Exception as e:
            logger.error(f"Error in step {i_step}: {str(e)}")
            raise

    def _handle_classification_task(self, new_samples: List[str], **kwargs) -> None:
        """Process classification task samples"""
        # Get annotations
        new_annotations = self._process_parallel(
            items_to_process=new_samples,
            process_func=self._annotate,
            llm=self.config.annotate_llm,
        )
        self.annotations.extend(new_annotations)

        # Get predictions
        new_predictions = self._process_parallel(
            items_to_process=new_samples,
            process_func=self._predict,
            llm=self.config.predictor_llm,
        )
        self.predictions.extend(new_predictions)

        # Evaluate and analyze results
        history = self._evaluate_and_analyze(
            inputs=new_samples,
            references=new_annotations,
            predictions=new_predictions,
        )
        self.history.append(history)

    def _format_prompts_by_label(
        self,
        history: List[StepResult] | None = None,
        extra_sample_str: str | None = None,
    ) -> List[str]:
        generate_prompts = []
        samples_per_label = self.config.samples_per_step // len(
            self.config.label_schema
        )
        attr = "step_adv_sample" if history else "adv_sample"
        prompt: str = getattr(self.prompt_handler, attr)

        for label in self.config.label_schema:
            generate_input = {
                "task_description": self.config.task_description,
                "label_schema": self.config.label_schema,
                "label": label,
                "instruction": self.cur_instruction,
            }
            if history:
                generate_input["history"] = history
                generate_input["extra_samples"] = extra_sample_str

            generate_prompts += [prompt.format_map(generate_input)] * samples_per_label

        remaining_labels = random.choices(
            self.config.label_schema,
            k=self.config.samples_per_step - len(generate_prompts),
        )
        for label in remaining_labels:
            generate_input = {
                "task_description": self.config.task_description,
                "label_schema": self.config.label_schema,
                "label": label,
                "instruction": self.cur_instruction,
            }
            if history:
                generate_input["history"] = history
                generate_input["extra_samples"] = extra_sample_str
            generate_prompts.append(prompt.format_map(generate_input))
        return generate_prompts

    def _handle_empty_samples(self) -> List[str]:
        """
        Handle case when no samples are available by generating initial samples.
        """
        generate_prompts = self._format_prompts_by_label()
        try:
            samples = [
                x.message.content
                for x in self._process_parallel(
                    items_to_process=generate_prompts,
                    process_func=self.config.generation_llm.chat,
                )
            ]

            logger.info(samples)
            logger.info(f"Generated {len(samples)} initial samples")
            return samples

        except Exception as e:
            logger.error(f"Failed to generate initial samples: {str(e)}")
            raise

    def _generate_adv_samples(self) -> List[str]:
        """
        Generate targeted samples based on error analysis after warmup phase.
        """
        indices = random.choices(
            range(len(self.samples)), k=self.config.num_extra_sample
        )
        extra_samples_text = ""
        for sample in [self.samples[i] for i in indices]:
            extra_samples_text += f"{sample}\n"
        history = self._format_history(self.history[-self.config.history_length :])
        generate_prompts = self._format_prompts_by_label(
            history=history, extra_sample_str=extra_samples_text
        )
        try:
            samples = [
                x.message.content
                for x in self._process_parallel(
                    items_to_process=generate_prompts,
                    process_func=self.config.generation_llm.chat,
                )
            ]
            logger.info(samples)
            logger.info(f"Generated {len(samples)} samples")
            return samples

        except Exception as e:
            logger.error(f"Failed to generate initial samples: {str(e)}")
            raise

    def _handle_generation_task(self, new_samples: List[str]) -> None:
        """
        Process generation task samples.
        """
        try:
            # Get predictor outputs
            new_predictions = self._predict(
                samples=new_samples, llm=self.config.predictor_llm
            )
            self.predictions.extend(new_predictions)

            # Get analyzer outputs
            class AnalysisSchema(BaseModel):
                quality: float = Field(...)
                feedback: str = Field(...)

            analyses = []
            for sample, prediction in zip(new_samples, new_predictions):
                analysis_input = {
                    "sample": sample,
                    "prediction": prediction,
                    "task_description": self.config.task_description,
                }

                analysis_prompt = self.prompt_handler.analyze_generation.format_map(
                    analysis_input
                )

                analysis = self.config.analyzer_llm.structured_output(
                    messages=analysis_prompt, schema=AnalysisSchema
                )
                analyses.append(analysis)

            # Calculate average quality score
            avg_quality = sum(a.quality for a in analyses) / len(analyses)

            # Generate consolidated feedback
            feedback_summary = self._consolidate_feedback(
                [a.feedback for a in analyses]
            )

            # Update history
            self.history.append(
                {
                    "score": avg_quality,
                    "analysis": feedback_summary,
                    "instruction": self.instruction,
                }
            )

            # Update best prompt if needed
            self.update_best_prompt(self.instruction, avg_quality)

        except Exception as e:
            logger.error(f"Error in generation task handling: {str(e)}")
            raise

    def _format_history(self, history: List[StepResult]) -> str:
        """
        Format history entries for prompt generation.
        """
        formatted = []
        for entry in history:
            formatted.append(
                f"Instruction: {entry.instruction}\n"
                f"Score: {entry.score:.2f}\n"
                f"Analysis: {entry.error_analysis.summary}"
            )
        return "\n\n".join(formatted)

    def _predict(self, sample: str, llm: BaseLLM) -> str:
        """Make predictions using specified LLM"""
        prompt_input = {
            "sample": sample,
            "instruction": self.cur_instruction,
            "label_schema": "".join(self.config.label_schema),
        }
        predict_prompt = self.prompt_handler.predict_single.format_map(prompt_input)
        # batch_results = llm.structured_output(
        #     messages=predict_prompt,
        #     schema=PredictsSchema
        # ).predictions
        result = llm.chat(messages=predict_prompt).message.content

        return result

    def _annotate(self, sample: str, llm: BaseLLM) -> str:
        """Make annotations using specified LLM"""
        return self._predict(sample=sample, llm=llm)

    def _evaluate_and_analyze(
        self, inputs: List[str], predictions: List[str], references: List[str]
    ) -> StepResult:
        print(predictions, references)
        """Evaluate and analyze classification results"""
        scores = [
            self.evaluator.evaluate_strings(prediction=p, reference=r)
            for p, r in zip(predictions, references)
        ]
        accuracy = len([s for s in scores if not self._error_func(s)]) / len(scores)

        # Generate error analysis
        error_analysis = self._generate_error_analysis(
            scores=scores,
            accuracy=accuracy,
            predictions=predictions,
            references=references,
            inputs=inputs,
        )
        return StepResult(
            score=accuracy,
            error_analysis=error_analysis,
            instruction=self.cur_instruction,
        )

    def _update_prompt(self, history: List[StepResult]) -> None:
        """Update prompt based on evaluation history"""
        if len(history) < self.config.warmup or len(history) % 3 > 0:
            last_history = history[-self.config.history_length :]
        else:
            sorted_history = sorted(
                history[self.config.warmup - 1 :],
                key=lambda x: x["score"],
                reverse=False,
            )
            last_history = sorted_history[-self.config.history_length :]
        prompt_input = {
            "history": self._format_history(last_history),
            "task_description": self.config.task_description,
            "error_analysis": last_history[-1].error_analysis.model_dump(),
            "labels": self.config.label_schema,
        }

        generate_prompt = (
            self.prompt_handler.update_prompt_classification
            if self.config.task_type == "classification"
            else self.prompt_handler.update_prompt_generation
        ).format_map(prompt_input)

        new_prompt = self.config.generation_llm.chat(
            messages=generate_prompt
        ).message.content

        self.add_metric(
            {"prompt": self.cur_instruction, "score": last_history[-1].score}
        )
        self.cur_instruction = new_prompt

    def stop_criteria(self) -> bool:
        """Check if optimization should stop"""
        if len(self.history) <= self.config.warmup:
            self.patient = 0
            return False

        if self.best_score - self.history[-1].score > -self.config.min_delta:
            self.patient += 1
        else:
            self.patient = 0
            self.best_score = self.history[-1].score
            self.best_instruction = self.history[-1].instruction

        return self.patient > self.config.patience

    def _after_run(self) -> Dict[str, Any]:
        """Finalize optimization and save results"""
        output_path = self.create_dated_directory(Path(self.config.store_path))
        self.save_state(output_path / "final_state.pkl")

        best_prompt = self.extract_best_prompt()
        return {
            "best_prompt": best_prompt,
            "best_score": self.best_score,
            "history": self.history,
            "metrics": self.metrics_history,
        }

    def _error_func(self, score):
        return score["score"] != 1

    def _generate_error_analysis(
        self,
        *,
        scores: List[Dict[str, float]],
        accuracy: float,
        predictions: List[str],
        references: List[str],
        inputs: List[str],
    ) -> ErrorAnalysis:
        """
        Generate detailed error analysis for classification results.

        Args:
            scores: List of evaluation scores for each prediction
            predictions: List of model predictions
            references: List of reference/ground truth labels
            inputs: List of input texts

        Returns:
            ErrorAnalysis Pydantic Model:
                - error_cases: ErrorCase clustered by label
                - summary: Text summary of error patterns
                - error_distribution: error count by label
        """
        # Collect error cases
        error_cases: List[ErrorCase] = []
        for i, (score, pred, ref, inp) in enumerate(
            zip(scores, predictions, references, inputs)
        ):
            if self._error_func(score):  # If prediction is wrong
                error_cases.append(ErrorCase(input=inp, prediction=pred, reference=ref))

        if not error_cases:
            return ErrorAnalysis(
                summary="No errors found",
                error_cases={key: [] for key in self.config.label_schema},
                error_distribution={key: 0 for key in self.config.label_schema},
            )

        # Prepare error_cases by label
        error_cases_by_labels = self.cluster_error_by_label(error_cases=error_cases)
        selected_error_cases: List[ErrorCase] = []
        for error_cases_by_label in error_cases_by_labels.values():
            selected_error_cases.extend(
                random.choices(error_cases_by_label, k=self.config.num_errors_per_label)
            )

        conf_matrix = confusion_matrix(
            y_true=references,
            y_pred=predictions,
            labels=self.config.label_schema,
        )
        conf_text = (
            f"Confusion matrix columns:{self.config.label_schema} the matrix data:"
        )
        for i, row in enumerate(conf_matrix):
            conf_text += f"\n{self.config.label_schema[i]}: {row}"

        # Prepare input for error analysis
        analysis_input = {
            "failure_cases": [
                error_case.model_dump() for error_case in selected_error_cases
            ],
            "task_description": self.config.task_description,
            "label_schema": self.config.label_schema,
            "instruction": self.cur_instruction,
            "score": accuracy,
            "confusion_matrix": conf_text,
        }

        try:
            # Generate structured analysis using analyzer LLM
            analysis_prompt = self.prompt_handler.error_analysis.format_map(
                analysis_input
            )
            analysis = self.config.analyzer_llm.chat(
                messages=analysis_prompt
            ).message.content

            error_distribution = dict(Counter(case.reference for case in error_cases))
            # Combine structured analysis with statistics
            return ErrorAnalysis(
                summary=analysis,
                error_cases=error_cases_by_labels,
                error_distribution=error_distribution,
            )

        except Exception as e:
            logger.error(f"Error in generating error analysis: {str(e)}")

    @staticmethod
    def cluster_error_by_label(
        error_cases: List[ErrorCase],
    ) -> Dict[str, List[ErrorCase]]:
        """
        Cluster items by attribute using groupby.
        Note: Items must be sorted by the attribute first.
        """
        # Sort items by the attribute first (required for groupby)
        sorted_items = sorted(error_cases, key=attrgetter("reference"))
        # Group items
        return {
            key: list(group)
            for key, group in groupby(sorted_items, key=attrgetter("reference"))
        }
