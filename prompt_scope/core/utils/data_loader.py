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
    delimiter: str = Field(default=',', description="delimiter for the csv file")
    quotechar: str = Field(default='"', description="quotechar for the csv file")

    def load_data(self) -> List[Dict[str, Any]]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f, delimiter=self.delimiter, quotechar=self.quotechar)
            return list(csv_reader)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f, delimiter=self.delimiter, quotechar=self.quotechar)
            yield from csv_reader


class HfDataLoader(BaseDataLoader):
    split: str = Field(default='train', description="delimiter for the csv file")
    subset: Optional[str] = Field(default=None, description="quotechar for the csv file")

    def __init__(self, dataset_name: str, split: str = 'train', subset: Optional[str] = None):
        self.split = split
        self.subset = subset

    def load_data(self) -> List[Dict[str, Any]]:
        dataset = load_dataset(self.file_path, name=self.subset, split=self.split)
        return list(dataset)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        dataset = load_dataset(self.file_path, name=self.subset, split=self.split)
        yield from dataset

class BBHDataLoader(BaseDataLoader):
    def load_data(self) -> List[Dict[str, Any]]:
        with open(self.file_path, 'r') as file:
            json_data = json.load(file)
            self.data = json_data.get('example', [])
        return self.data

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self.load_data()
        return iter(self.data)


class AQuADataLoader(JsonlDataLoader):
    pass


class GSMDataLoader(CsvDataLoader):
    pass


class MMLUDataLoader(CsvDataLoader):
    pass


class MultiArithDataLoader(BaseDataLoader):
    def load_data(self) -> List[Dict[str, Any]]:
        with open(self.file_path, 'r') as f:
            self.data = json.load(f)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self.load_data()
        for item in self.data:
            yield item