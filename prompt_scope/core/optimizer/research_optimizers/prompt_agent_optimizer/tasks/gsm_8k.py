# define task prompts for various datasets
import re

from .base_task import BaseDataset, BaseTask
from prompt_scope.datasets.data_loader import GSMDataLoader
import random

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

                 task_name="gsm_8k",
                 task_description="gsm_8k",
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
        self.answer_format_prompt = ""

    def load_task_dataset(self, data_dir):
        '''
            <task specific>
        '''
        csv_data = GSMDataLoader(
            file_path=data_dir,
            index_col=None,
            header=None,
            sep="\t").load_data()
        return csv_data

    def transform_format(self, data):
        questions = data[0]
        answers = data[1]

        examples = []
        # Extracting input and target scores
        for question, answer in zip(questions, answers):
            formatted_example = {
                'question': question,
                'answer': str(answer)
            }
            examples.append(formatted_example)
        return examples

    def clean_response(self, response):
        integer_pattern = r"\d+"
        matches = re.findall(integer_pattern, response)
        if len(matches) != 0:
            return str(matches[-1])
        extended_pattern = r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one)\b"
        matches = re.findall(extended_pattern, response)
        if len(matches) != 0:
            return str(number_to_word_dict[matches[-1]])
        else:
            return "N/A: format error."
        
    def split_list_dataset(self, dataset, train_size=None, eval_size=150, test_size=0, seed=None, base_shuffle=True):
        if base_shuffle and seed is not None:
            if seed is not None:
                print(f'shuffle dataset seed {seed}')
                random.seed(seed)
            random.shuffle(dataset)

        test_set = dataset[:test_size]
        dataset = dataset[test_size:]

        if train_size is not None:
            train_set = dataset[:train_size]
        else:
            train_set = dataset
        eval_size = max(50, min(eval_size, 200))
        eval_set = dataset[-eval_size:]

        return train_set, eval_set, test_set

