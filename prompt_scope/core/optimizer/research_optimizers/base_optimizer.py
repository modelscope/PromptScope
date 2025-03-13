from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime
from loguru import logger
from contextlib import contextmanager
import time
import pickle as pkl
import os
from tqdm import tqdm
import concurrent.futures

from prompt_scope.core.utils.prompt_handler import PromptHandler
from prompt_scope.core.evals.loading import load_evaluator
from prompt_scope.core.evals.schema import StringEvaluator, EvaluatorType
from prompt_scope.core.utils.logging_utils import LoggerFactory


class OptimizationConfig(BaseModel):
    """Base configuration for prompt optimization"""

    store_path: Path
    num_steps: int
    batch_size: int = 1
    verbose: bool = False
    seed: int = 42
    eval_type: EvaluatorType
    language: str = "en"
    prompt_path: Path
    max_workers: int = 4
    init_instruction: str
    patience: int = 10


class PromptOptimizationWithFeedback(ABC):
    """
    Abstract base class for prompt optimization algorithms with feedback.

    This class provides a framework for implementing prompt optimization algorithms
    that use feedback loops to improve prompt quality. It supports both synchronous
    and asynchronous evaluation, metric tracking, and state management.
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.metrics_history = []
        self.cur_instruction = self.config.init_instruction  # Current prompt being optimized
        self.best_instruction = self.config.init_instruction  # Best prompt found so far
        self.best_score = float("-inf")
        self.cur_step = 0
        self._prompt_handler = None
        self._evaluator = None
        logger_factory = LoggerFactory(log_dir=self.config.store_path)
        self.logger = logger_factory.get_logger("my_module")

    @property
    def evaluator(self) -> StringEvaluator:
        if self.config.eval_type == "custom":
            raise ValueError("evaluator is not supported when eval_type is custom")
        if not self._evaluator:
            self._evaluator = load_evaluator(self.config.eval_type)
        return self._evaluator

    @property
    def prompt_handler(self) -> PromptHandler:
        """
        Returns:
            PromptHandler: An instance of PromptHandler initialized with specific file path and keyword arguments.
        """
        if not self._prompt_handler:
            self._prompt_handler = PromptHandler(
                self.config.prompt_path, language=self.config.language
            )
        return self._prompt_handler

    @contextmanager
    def _step_logger(self, step: int):
        """Context manager for logging step execution"""
        start_time = time.time()
        logger.info(f"Starting step {step}")
        try:
            yield
        finally:
            duration = time.time() - start_time
            logger.info(f"Completed step {step} in {duration:.2f}s")

    def run(self) -> Dict[str, Any]:
        """
        Main optimization loop.

        Returns:
            Dict containing optimization results and best prompt
        """
        try:
            # Initialize optimization
            data_kwargs, result_kwargs = self._before_run()

            # Main optimization loop
            for i_step in range(self.config.num_steps):
                with self._step_logger(i_step):
                    should_stop = self._step(
                        i_step=i_step, **data_kwargs, **result_kwargs
                    )
                    if should_stop:
                        logger.info("Early stopping criteria met")
                        break
                    self.cur_step += 1
            # Finalize and return results
            return self._after_run()

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise

    @abstractmethod
    def _before_run(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Initialize optimization process.

        Returns:
            Tuple of (data_kwargs, result_kwargs) for use in optimization steps
        """
        pass

    @abstractmethod
    def _step(self, *, i_step: int, **kwargs) -> bool:
        """
        Execute single optimization step.

        Args:
            i_step: Current step number
            **kwargs: Additional arguments needed for step execution

        Returns:
            bool: True if optimization should stop, False otherwise
        """
        pass

    @abstractmethod
    def _after_run(self) -> Dict[str, Any]:
        """
        Finalize optimization process.

        Returns:
            Dict containing final results
        """
        pass

    @abstractmethod
    def _predict(self, **kwargs) -> Any:
        """Make predictions using current prompt"""
        pass

    @abstractmethod
    def _evaluate_and_analyze(
        self, inputs: List[str], predictions: List[str], references: List[str]
    ) -> Any:
        """Evaluate predictions and analyze errors"""
        pass

    @abstractmethod
    def _update_prompt(self, **kwargs) -> Any:
        """Update prompt based on evaluation results"""
        pass

    def save_state(self, path: Path) -> None:
        """Save current optimization state"""
        state = {
            "cur_step": self.cur_step,
            "instruction": self.cur_instruction,
            "best_instruction": self.best_instruction,
            "best_score": self.best_score,
            "metrics_history": self.metrics_history,
        }
        path.write_bytes(pkl.dumps(state))

    def load_state(self, path: Path) -> None:
        """Load optimization state"""
        if path.exists():
            state = pkl.loads(path.read_bytes())
            for key, value in state.items():
                setattr(self, key, value)

    def extract_best_prompt(self) -> str:
        """Return best prompt found during optimization"""
        return self.best_instruction

    @staticmethod
    def create_dated_directory(base_path: Path) -> Path:
        """Create dated directory for storing results"""
        current_date = datetime.now().strftime("%Y%m%d")
        version = 1

        while True:
            dir_name = f"{current_date}_v{version}"
            full_path = base_path / dir_name

            if not full_path.exists():
                full_path.mkdir(parents=True)
                return full_path

            version += 1

    def update_best_prompt(self, prompt: str, score: float) -> None:
        """Update best prompt if new score is better"""
        if score > self.best_score:
            self.best_score = score
            self.best_instruction = prompt
            logger.info(f"New best prompt found (score: {score:.4f})")

    def add_metric(self, metric: Dict[str, Any]) -> None:
        """Add metric to history"""
        metric["step"] = self.cur_step
        metric["timestamp"] = datetime.now().isoformat()
        self.metrics_history.append(metric)

    def _process_parallel(
        self, items_to_process, process_func, **process_func_kwargs
    ) -> None:
        """Execute the pipeline for all items with continuous parallel processing."""
        # 创建进度条
        pbar = tqdm(
            total=len(items_to_process),
            initial=0,
            desc="Processing items",
            position=0,
            leave=True,
        )

        results = [None] * len(items_to_process)

        def process_with_progress(idx_item):
            """处理单个项目并更新进度条"""
            idx, item = idx_item
            try:
                result = process_func(item, **process_func_kwargs)
                results[idx] = result  # Store result at the correct index
            except Exception as e:
                logger.exception(f"Error processing item at index {idx}: {str(e)}")
            finally:
                pbar.update(1)

        # 使用线程池持续处理所有数据
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers
        ) as executor:
            # 提交所有任务
            futures = [
                executor.submit(process_with_progress, (idx, item))
                for idx, item in enumerate(items_to_process)
            ]
            # 等待所有任务完成
            concurrent.futures.wait(futures)
        pbar.close()
        return results
