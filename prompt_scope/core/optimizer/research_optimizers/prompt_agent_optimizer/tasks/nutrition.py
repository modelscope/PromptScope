# define task prompts for various datasets
import re

from .base_task import BaseDataset, BaseTask

from prompt_scope.datasets.data_loader import CMMLUDataLoader

class CustomTask(BaseTask):
    def __init__(self,
                 train_size,
                 eval_size,
                 test_size=None,

                 task_name="nutrition",
                 task_description="nutrition",
                 data_dir='',
                 seed=None,

                 post_instruction=True,
                 TaskDataset=BaseDataset,
                 option_num=5,
                 **kwargs):
        self.options = {}
        super().__init__(
            task_name=task_name,
            task_description=task_description,
            data_dir=data_dir,
            seed=seed,
            train_size=train_size,
            eval_size=eval_size,
            test_size=test_size,
            post_instruction=post_instruction,
            TaskDataset=TaskDataset,
            option_num=option_num,
        )
        self.answer_format_prompt = "最终答案请按照下面格式给出：<answer>xx</answer>，请注意最终答案仅包含选项，即A, B, C, D中的一个"

    def load_task_dataset(self, data_dir):
        '''
            <task specific>
        '''
        data_list = CMMLUDataLoader(file_path=data_dir).load_data()
        return data_list

    def transform_format(self, data):

        examples = []
        # Extracting input and target scores
        for example in data:
            question = example['input']
            answer = example['output']
            formatted_example = {
                'question': question,
                'answer': answer
            }
            examples.append(formatted_example)
        return examples
    
    def clean_response(self, response):
        '''
        <task specific>
        Extract the answers from pred_model's response.
        '''
        clean_pattern = r'<answer>(.*?)</answer>'
        match = re.findall(clean_pattern, response)
        if len(match) == 0:
            return 'N/A: Format error'
        return match[-1]