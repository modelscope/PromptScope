from abc import ABC, abstractmethod
import json
import csv
from typing import List, Dict, Any, Optional, Iterator
from datasets import load_dataset
from pydantic import Field


class BaseDataLoader(ABC):
    file_path: str = Field(..., description="Path to the file containing the dataset")
    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        pass


class JsonlDataLoader(BaseDataLoader):
    def load_data(self) -> List[Dict[str, Any]]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line.strip()) for line in f]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line.strip())


class CsvDataLoader(BaseDataLoader):
    def __init__(self, file_path: str, delimiter: str = ',', quotechar: str = '"'):
        super().__init__(file_path)
        self.delimiter = delimiter
        self.quotechar = quotechar

    def load_data(self) -> List[Dict[str, Any]]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f, delimiter=self.delimiter, quotechar=self.quotechar)
            return list(csv_reader)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f, delimiter=self.delimiter, quotechar=self.quotechar)
            yield from csv_reader


class HfDataLoader(BaseDataLoader):
    def __init__(self, dataset_name: str, split: str = 'train', subset: Optional[str] = None):
        super().__init__(dataset_name)
        self.split = split
        self.subset = subset

    def load_data(self) -> List[Dict[str, Any]]:
        dataset = load_dataset(self.file_path, name=self.subset, split=self.split)
        return list(dataset)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        dataset = load_dataset(self.file_path, name=self.subset, split=self.split)
        yield from dataset



