from sklearn.metrics import confusion_matrix
import os

from meta_icl.core.models.generation_model import LlamaIndexGenerationModel
from meta_icl.core.utils.logger import Logger
from meta_icl.core.utils.ipc_config import load_yaml
from meta_icl import CONFIG_REGISTRY, PROMPT_REGISTRY
import random

class Eval:
    """
    The Eval class is responsible to calculate the score and the large errors
    """

    def __init__(self):
        """
        Initialize a new instance of the Eval class.
        :param config: The configuration file (EasyDict)
        :analyzer (optional): A chain that analyze the errors
        :label_schema (optional): The label schema
        """
        self.eval_config = CONFIG_REGISTRY.module_dict['eval_config']
        self.task_config = CONFIG_REGISTRY.module_dict['task_config']
        self.model_config = CONFIG_REGISTRY.module_dict['model_config']

        self.analyzer_llm = LlamaIndexGenerationModel(**self.model_config.analyzer)
        if hasattr(self.model_config, 'evaluator'):
            self.evaluator_llm = LlamaIndexGenerationModel(**self.model_config.evaluator)

        self.history = []        
        self.logger = Logger.get_logger(__name__)
        self.mean_score = None

    @staticmethod
    def eval_with_llm(samples):
        evaluation_prompt = PROMPT_REGISTRY.module_dict['annotate'].format(samples=sample_str, instruction=self.task_config.instruction, batch_size=self.task_config.batch_size)
        prediction_dict = {item['ID']: item['预测'] for item in predictions}
        total = 0
        correct = 0
        correct_sample, error_sample, anno_only, pred_only = [], [[] for _ in range(len(label_schema))], [], []
        item_dict = {item: index for index, item in enumerate(label_schema)}
        print(item_dict)
        for item in annotations:
            item_copy = item.copy()
            item_id = item['ID']
            annotation = item['标注']
            prediction = prediction_dict.get(item_id)
            item_copy['预测'] = prediction
            if prediction is not None:
                total += 1
                if prediction == annotation:
                    correct += 1
                    correct_sample.append(item_copy)
                else:
                    error_sample[item_dict[annotation]].append(item_copy)
            anno_only.append(annotation)
            pred_only.append(prediction)
        
        accuracy = correct / total if total > 0 else 0
        conf_matrix = confusion_matrix(anno_only, pred_only, labels=label_schema)
        return accuracy, correct_sample, error_sample, conf_matrix
    
    @staticmethod
    def eval_accuracy(annotations, predictions, label_schema):
        prediction_dict = {item['ID']: item['预测'] for item in predictions}
        total = 0
        correct = 0
        correct_sample, error_sample, anno_only, pred_only = [], [[] for _ in range(len(label_schema))], [], []
        item_dict = {item: index for index, item in enumerate(label_schema)}
        print(item_dict)
        for item in annotations:
            item_copy = item.copy()
            item_id = item['ID']
            annotation = item['标注']
            prediction = prediction_dict.get(item_id)
            item_copy['预测'] = prediction
            if prediction is not None:
                total += 1
                if prediction == annotation:
                    correct += 1
                    correct_sample.append(item_copy)
                else:
                    error_sample[item_dict[annotation]].append(item_copy)
            anno_only.append(annotation)
            pred_only.append(prediction)
        
        accuracy = correct / total if total > 0 else 0
        conf_matrix = confusion_matrix(anno_only, pred_only, labels=label_schema)
        return accuracy, correct_sample, error_sample, conf_matrix
    
    def error_to_str(self, errors, num_errors_per_label):
        txt_res = ''
        if self.eval_config.func_name == 'accuracy':
            gt_name = 'GT'
        else:
            gt_name = 'rank'
        for error_labels in errors:
            random.shuffle(error_labels)
            for error in error_labels[:num_errors_per_label]:
                txt_res += f"Sample: {error['问题']}\nPrediction: {error['预测']}, {gt_name}: {error['标注']}\n#\n"
        return txt_res

    def error_analysis(self, prompt, annotations, predictions):
        error_str = self.error_to_str(self.errors, num_errors_per_label=self.eval_config.num_errors_per_label)
        prompt_input = {'task_description': self.task_config.task_description, 'accuracy': self.mean_score, 'prompt': prompt, 'failure_cases': error_str}
        if self.eval_config.func_name == 'accuracy':
            conf_text = f"Confusion matrix columns:{self.eval_config.label_schema} the matrix data:"
            for i, row in enumerate(self.conf_matrix):
                conf_text += f"\n{self.eval_config.label_schema[i]}: {row}"
            prompt_input['confusion_matrix'] = conf_text
        elif self.eval_config.func_name == 'ranking':
            prompt_input['labels'] = self.eval_config.label_schema

        self.logger.info(prompt_input)

        analyze_prompt = PROMPT_REGISTRY.module_dict['error_analysis'].format_map(prompt_input)
        analysis = self.analyzer_llm.call(prompt=analyze_prompt)
        self.logger.info(analysis.message.content)
        self.history.append({'prompt': prompt, 'score': self.mean_score,
                             'errors': self.errors, 'confusion_matrix': conf_matrix, 'analysis': analysis.message.content})
        
    def sample_to_text(self, sample: dict, num_errors_per_label: int = 0, is_score: bool = True) -> str:
        """
        Return a string that organize the information of from the step run for the meta-prompt
        :param sample: The eval information for specific step
        :param num_errors_per_label: The max number of large errors per class that will appear in the meta-prompt
        :param is_score: If True, add the score information to the meta-prompt
        :return: A string that contains the information of the step run
        """
        if is_score:
            return f"####\n##Prompt Score: {sample['score']:.2f}\n##Prompt:\n{sample['prompt']}\n#################\n"
        else:
            return f"####\n##Prompt:\n{sample['prompt']}\n{self.error_to_str(sample['errors'], num_errors_per_label)}####\n "
        
    def get_max_score(self, warmup=0):
        """
        Return the maximum 'mean score' (with respect to all history epochs, starting form warmup, up to last) and the epoch index of the maximum score
        :return: The epoch index of the maximum score, and the maximum score
        """
        return max(self.history[warmup-1:-1], key=lambda x:x['score'])['score']