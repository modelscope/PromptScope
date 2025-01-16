# define task prompts for various datasets
import re

from .base_task import BaseDataset, BaseTask

number_to_word_dict = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "twenty-one": 21
}


class CustomTask(BaseTask):
    def __init__(self,
                 train_size,
                 eval_size,
                 test_size=None,

                 task_name="word_sorting",
                 task_description="word_sorting",
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
        self.answer_format_prompt = "At the end show the answer option bracketed between <answer> and </answer>."

    def load_task_dataset(self, data_dir):
        '''
            <task specific>
        '''
        json_data = self._load_json_file(data_dir)
        return json_data

    def transform_format(self, data):
        original_examples = data['examples']

        examples = []
        # Extracting input and target scores
        for example in original_examples:
            question = example['input']
            answer = example['target']
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
        match = re.findall(clean_pattern, response.lower())
        if len(match) == 0:
            return 'N/A: Format error'
        return match[-1]