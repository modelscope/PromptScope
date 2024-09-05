import unittest
from unittest.mock import MagicMock, patch, PropertyMock, Mock
from meta_icl.core.offline.instruction_optimization.ipc import IPCOptimization
from meta_icl.core.offline.instruction_optimization.opro import OPRO
from meta_icl.core.offline.instruction_optimization.prompt_agent import PromptAgent
from meta_icl.core.utils.utils import load_yaml
from meta_icl import CONFIG_REGISTRY
from meta_icl.core.evaluation.evaluator import Eval
from meta_icl.algorithm.PromptAgent.search_algo.mcts import MCTS
from meta_icl.algorithm.PromptAgent.search_algo.beam_search import BeamSearch
from meta_icl.core.models.world_model import WorldModel
from meta_icl.core.models.beam_world_model import BeamSearchWorldModel



# class FlexiMock(MagicMock):
#     def __init__(self, *args, **kwargs):
#         super(FlexiMock, self).__init__(*args, **kwargs)
#         self.__getitem__ = lambda obj, item: getattr(obj, item) # set the mock for `[...]`

#     def get(self, x, default=None):
#         return getattr(self, x, default)
    
class TestConfigRegistry(unittest.TestCase):
    def setUp(self):
        pass       

    @patch('meta_icl.algorithm.base_algorithm.PromptOptimizationWithFeedback.__init__')
    def test_ipc(self, mock_super_init: MagicMock):
        mock_config = load_yaml('configs/ipc_optim_generate.yml')
        CONFIG_REGISTRY.batch_register(mock_config)
        # 设置 IPCOptimization 实例
        language = "cn"
        ipc_optimization = IPCOptimization(language=language)

        # 验证是否调用了超类的初始化方法
        mock_super_init.assert_called_once_with(language=language)

        # 验证关键属性是否正确初始化
        self.assertEqual(ipc_optimization.patient, 0)
        self.assertIsNone(ipc_optimization.samples)
        self.assertEqual(ipc_optimization.cur_step, 0)
        self.assertEqual(ipc_optimization.cur_prompt, mock_config.task_config.instruction)
        self.assertIsInstance(ipc_optimization.eval, Eval)

    @patch('meta_icl.algorithm.base_algorithm.PromptOptimizationWithFeedback.__init__')
    def test_opro(self, mock_super_init: MagicMock):
        # 测试默认情况下的run方法
        mock_config = load_yaml('configs/opro.yml')
        CONFIG_REGISTRY.batch_register(mock_config)
        language = "cn"
        opro = OPRO(language=language)

        mock_super_init.assert_called_once_with(language=language)

        self.assertIn(opro.dataset_name, {
            "mmlu",
            "bbh",
            "gsm8k",
        })
        if opro.dataset_name == "mmlu":
            self.assertIn(opro.task_name, {
                "STEM",
                "humanities",
                "social sciences",
                "other (business, health, misc.)",
            })
        elif opro.dataset_name == "bbh":
            self.assertIn(opro.task_name, {
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
            })
        else:
            self.assertEqual(opro.dataset_name, "gsm8k")
            self.assertIn(opro.task_name, {"train", "test"})

        self.assertIn(opro.meta_prompt_type, {
            "both_instructions_and_exemplars",
            "instructions_only",
        })

        self.assertIn(opro.instruction_pos, {
            "before_Q",
            "Q_begin",
            "Q_end",
            "A_begin",
        })
    @patch('meta_icl.algorithm.PromptAgent.tasks.bigbench.CustomTask')
    def test_prompt_agent(self, mock_get_task: MagicMock):
        # 测试默认情况下的run方法
        mock_config = load_yaml('configs/prompt_agent.yml')
        CONFIG_REGISTRY.batch_register(mock_config)
        language = "cn"

        mock_task = MagicMock()
        mock_get_task.return_value = mock_task
        prompt_agent = PromptAgent(language=language)
        
        self.assertIsInstance(prompt_agent.dataset_path, str)
        self.assertIsInstance(prompt_agent.task_config.data_dir, str)
        self.assertIsInstance(prompt_agent.initial_prompt, str)
        mcts = isinstance(prompt_agent.search_algo, MCTS) and isinstance(prompt_agent.world_model, WorldModel)
        beam_search = isinstance(prompt_agent.search_algo, BeamSearch) and isinstance(prompt_agent.world_model, BeamSearchWorldModel)
        self.assertIs(mcts or beam_search, True)


# 运行测试
if __name__ == '__main__':
    unittest.main()