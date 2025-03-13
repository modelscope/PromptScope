import unittest
from unittest.mock import Mock, patch, MagicMock, create_autospec
import pytest
from pathlib import Path
import tempfile

from prompt_scope.core.optimizer.research_optimizers.base_optimizer import (
    OptimizationConfig,
    PromptOptimizationWithFeedback,
)
from prompt_scope.core.optimizer.research_optimizers.ipc_optimizer.ipc import (
    IPCOptimization,
    IPCConfig,
    ErrorAnalysis,
    StepResult,
    ErrorCase,
)

class Message:
    def __init__(self):
        self.content = "Error analysis summary"

class AnalyzerResponse:
    def __init__(self):
        self.message = Message()

# Base class tests
class TestPromptOptimizationWithFeedback(unittest.TestCase):
    """Test cases for the base optimization class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = OptimizationConfig(
            store_path=Path("/tmp/test"),
            num_steps=10,
            batch_size=2,
            verbose=True,
            seed=42,
            eval_type="exact_match",
            init_instruction="my_prompt",
            prompt_path="my_path"
        )

        # Create concrete implementation for testing
        class ConcreteOptimizer(PromptOptimizationWithFeedback):
            def _before_run(self):
                return {}, {}

            def _step(self, i_step, **kwargs):
                return False

            def _after_run(self):
                return {"result": "test"}

            def _predict(self, **kwargs):
                return []

            def _evaluate_and_analyze(self, **kwargs):
                return []

            def _update_prompt(self, **kwargs):
                pass

        self.optimizer = ConcreteOptimizer(self.config)

    def test_initialization(self):
        """Test proper initialization of optimizer"""
        self.assertEqual(self.optimizer.cur_step, 0)
        self.assertEqual(self.optimizer.best_score, float("-inf"))
        self.assertEqual(self.optimizer.metrics_history, [])

    def test_run_execution(self):
        """Test main optimization loop execution"""
        result = self.optimizer.run()
        self.assertEqual(result, {"result": "test"})

    def test_save_load_state(self):
        """Test state saving and loading"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.pkl"

            # Set some state
            self.optimizer.best_score = 0.8
            self.optimizer.best_instruction = "test instruction"

            # Save state
            self.optimizer.save_state(path)

            # Create new optimizer and load state
            new_optimizer = type(self.optimizer)(self.config)
            new_optimizer.load_state(path)

            self.assertEqual(new_optimizer.best_score, 0.8)
            self.assertEqual(new_optimizer.best_instruction, "test instruction")

    def test_metric_tracking(self):
        """Test metric tracking functionality"""
        metric = {"score": 0.8, "loss": 0.2}
        self.optimizer.add_metric(metric)

        self.assertEqual(len(self.optimizer.metrics_history), 1)
        self.assertIn("timestamp", self.optimizer.metrics_history[0])
        self.assertEqual(self.optimizer.metrics_history[0]["score"], 0.8)

    def test_best_prompt_update(self):
        """Test best prompt updating logic"""
        self.optimizer.update_best_prompt("test prompt", 0.8)
        self.assertEqual(self.optimizer.best_score, 0.8)
        self.assertEqual(self.optimizer.best_instruction, "test prompt")

        # Should not update for worse score
        self.optimizer.update_best_prompt("worse prompt", 0.7)
        self.assertEqual(self.optimizer.best_instruction, "test prompt")


class TestIPCOptimization(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        # Create mock config
        self.mock_config = IPCConfig(
            task_type="classification",
            label_schema=["positive", "negative", "neutral"],
            task_description="Sentiment analysis task",
            samples_per_step=10,
            max_samples=50,
            warmup=4,
            history_length=4,
            num_errors_per_label=5,
            eval_type='exact_match',
            init_instruction="Initial instruction",
            num_steps=10,
        )

        # Create mock LLMs
        self.mock_llms = {
            "generation_llm": Mock(),
            "predictor_llm": Mock(),
            "analyzer_llm": Mock(),
            "annotate_llm": Mock(),
        }
        for name, mock_llm in self.mock_llms.items():
            setattr(self.mock_config, name, mock_llm)

        # Create optimizer instance
        self.optimizer = IPCOptimization(self.mock_config)

    def test_initialization(self):
        """Test proper initialization of IPCOptimization"""
        self.assertEqual(self.optimizer.config.task_type, "classification")
        self.assertEqual(len(self.optimizer.config.label_schema), 3)
        self.assertEqual(self.optimizer.patient, 0)
        self.assertEqual(len(self.optimizer.history), 0)

    def test_handle_empty_samples(self):
        """Test handling of empty samples"""
        # Mock generation LLM response
        mock_response = Mock()
        mock_response.message.content = "Generated sample text"
        self.mock_config.generation_llm.chat.return_value = mock_response

        samples = self.optimizer._handle_empty_samples()

        # Verify results
        self.assertEqual(len(samples), self.mock_config.samples_per_step)
        self.mock_config.generation_llm.chat.assert_called()

    def test_generate_adv_samples(self):
        """Test generation of adversarial samples"""
        # Setup test data
        self.optimizer.samples = ["sample1", "sample2", "sample3"]
        self.optimizer.history = [
            StepResult(
                score=0.8,
                error_analysis=ErrorAnalysis(
                    summary="Test summary", error_cases={}, error_distribution={}
                ),
                instruction="Test instruction",
            )
        ] * 5

        # Mock generation LLM response
        mock_response = Mock()
        mock_response.message.content = "Generated adversarial sample"
        self.mock_config.generation_llm.chat.return_value = mock_response

        samples = self.optimizer._generate_adv_samples()

        # Verify results
        self.assertEqual(len(samples), self.mock_config.samples_per_step)
        self.mock_config.generation_llm.chat.assert_called()

    def test_handle_classification_task(self):
        """Test classification task handling"""
        # Setup test data
        new_samples = ["sample1", "sample2"]

        # Mock responses
        mock_annotation = Mock()
        mock_annotation.message.content = "positive"
        self.mock_config.annotate_llm.chat.return_value = mock_annotation

        mock_prediction = Mock()
        mock_prediction.message.content = "negative"
        self.mock_config.predictor_llm.chat.return_value = mock_prediction

        # Execute
        self.optimizer._handle_classification_task(
            new_samples=new_samples,
        )

        # Verify
        self.assertEqual(len(self.optimizer.annotations), len(new_samples))
        self.assertEqual(len(self.optimizer.predictions), len(new_samples))

    def test_error_analysis(self):
        """Test error analysis generation"""
        # Setup test data
        scores = [{"score": 0}, {"score": 1}]
        predictions = ["positive", "negative"]
        references = ["negative", "negative"]
        inputs = ["text1", "text2"]

        # Mock analyzer response
        self.mock_config.analyzer_llm.chat.return_value = AnalyzerResponse
        
        summary = self.mock_config.analyzer_llm.chat(messages="please return a string")
        print(summary)
        self.assertIsInstance(summary, str)

        analysis = self.optimizer._generate_error_analysis(
            scores=scores,
            accuracy=0.5,
            predictions=predictions,
            references=references,
            inputs=inputs,
        )

        # Verify results
        self.assertIsInstance(analysis, ErrorAnalysis)
        self.assertTrue(analysis.error_cases)
        self.assertTrue(analysis.error_distribution)
        self.assertIsInstance(analysis.summary, str)

    def test_update_prompt(self):
        """Test prompt updating"""
        # Setup test history
        history = [
            StepResult(
                score=0.8,
                error_analysis=ErrorAnalysis(
                    summary="Test summary", error_cases={}, error_distribution={}
                ),
                instruction="Test instruction",
            )
        ] * 5

        # Mock LLM response
        mock_response = Mock()
        mock_response.message.content = "Updated prompt"
        self.mock_config.generation_llm.chat.return_value = mock_response

        # Execute
        self.optimizer._update_prompt(history)

        # Verify
        self.assertEqual(self.optimizer.cur_instruction, "Updated prompt")
        self.mock_config.generation_llm.chat.assert_called()

    def test_stop_criteria(self):
        """Test stop criteria"""
        # Test with insufficient history
        self.assertFalse(self.optimizer.stop_criteria())

        # Test with sufficient history and improving scores
        self.optimizer.history = [
            StepResult(
                score=0.7,
                error_analysis=ErrorAnalysis(
                    summary="Test", error_cases={}, error_distribution={}
                ),
                instruction="Test",
            )
        ] * (self.mock_config.warmup + 1)
        self.assertFalse(self.optimizer.stop_criteria())

    def test_cluster_error_by_label(self):
        """Test error case clustering"""
        error_cases = [
            ErrorCase(input="text1", prediction="pos", reference="neg"),
            ErrorCase(input="text2", prediction="neg", reference="neg"),
            ErrorCase(input="text3", prediction="pos", reference="pos"),
        ]

        clustered = self.optimizer.cluster_error_by_label(error_cases)

        self.assertIsInstance(clustered, dict)
        self.assertEqual(len(clustered), 2)  # Two unique reference labels
        self.assertTrue(all(isinstance(v, list) for v in clustered.values()))


if __name__ == "__main__":
    unittest.main()
