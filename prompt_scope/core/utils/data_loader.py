import json
from typing import List, Dict, Any
import csv
from datasets import load_dataset, Dataset

class BaseDataloader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load the data from the file.
        To be implemented by subclasses.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    def get_batch(self, batch_size: int, start_idx: int = 0) -> List[Dict[str, Any]]:
        end_idx = min(start_idx + batch_size, len(self.data))
        return self.data[start_idx:end_idx]

class JsonlDataloader(BaseDataloader):
    def load_data(self) -> Dataset:
        data = []
        with open(self.file_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                data.append({
                    'instruction': sample.get('instruction', ''),
                    'demonstration': sample.get('demonstration', []),
                    'query': sample.get('query', ''),
                    'output': sample.get('output', '')
                })
        return Dataset.from_list(data)

class CsvDataloader(BaseDataloader):
    def load_data(self) -> Dataset:
        data = []
        with open(self.file_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    'instruction': row.get('instruction', ''),
                    'demonstration': row.get('demonstration', '').split('|'),  # Assuming demonstrations are pipe-separated
                    'query': row.get('query', ''),
                    'output': row.get('output', '')
                })
        return Dataset.from_list(data)

class HfDataloader(BaseDataloader):
    def __init__(self, dataset_name: str, split: str = 'train'):
        self.dataset_name = dataset_name
        self.split = split
        super().__init__(file_path='')  # file_path is not used for HF datasets

    def load_data(self) -> Dataset:
        dataset = load_dataset(self.dataset_name, split=self.split)
        
        # Ensure the dataset has the required columns
        if not all(col in dataset.column_names for col in ['instruction', 'demonstration', 'query', 'output']):
            # If any column is missing, create it with default values
            dataset = dataset.map(
                lambda example: {
                    'instruction': example.get('instruction', ''),
                    'demonstration': example.get('demonstration', []),
                    'query': example.get('query', ''),
                    'output': example.get('output', '')
                }
            )
        
        return dataset