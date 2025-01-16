from abc import ABC, abstractmethod
import json
import csv
from typing import List, Dict, Any, Optional, Iterator, Sequence, Union, Literal
from datasets import load_dataset
from pydantic import Field, BaseModel
from enum import Enum
from pathlib import PosixPath
import pandas as pd

class BenchmarkName(Enum):
    BBH = "bbh"
    AQUA = "aqua"
    GSM = "gsm"
    MMLU = "mmlu"
    MULTIARITH = "multiarith"
    
class BaseDataLoader(BaseModel, ABC):
    file_path: Union[str, PosixPath] = Field(..., description="Path to the file containing the dataset")
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
    index_col: int | str | Sequence[str | int] | Literal[False] | None
    header: int | Sequence[int] | Literal["infer"] | None
    sep: str | Sequence[int] | Literal["infer"] | None
    
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(
            self.file_path,
            index_col=self.index_col,
            header=self.header,
            sep=self.sep,
            encoding='utf-8'
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # For memory efficiency, use chunksize
        chunks = pd.read_csv(
            self.file_path,
            index_col=self.index_col,
            header=self.header,
            sep=self.sep,
            encoding='utf-8',
            chunksize=1000  # Adjust chunk size as needed
        )
        for chunk in chunks:
            for record in chunk.to_dict('records'):
                yield record


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

# class DataLoaderRouter:
#     """Router class to select appropriate data loader based on dataset name"""
    
#     @staticmethod
#     def get_loader(dataset_name: str, datafile_name: str, **kwargs) -> Optional[BaseDataLoader]:
#         """
#         Get the appropriate data loader instance based on dataset name.
        
#         Args:
#             dataset_name: Name of the dataset (case-insensitive)
#             file_path: Path to the data file
#             **kwargs: Additional arguments to pass to the data loader
            
#         Returns:
#             An instance of the appropriate data loader
            
#         Raises:
#             ValueError: If dataset_name is not recognized
#         """
#         try:
#             dataset_enum = BenchmarkName(dataset_name.lower())
#         except ValueError:
#             raise ValueError(f"Unsupported dataset: {dataset_name}. "
#                            f"Supported datasets are: {[name.value for name in BenchmarkName]}")

#         loader_mapping = {
#             BenchmarkName.BBH: BBHDataLoader,
#             BenchmarkName.AQUA: AQuADataLoader,
#             BenchmarkName.GSM: GSMDataLoader,
#             BenchmarkName.MMLU: MMLUDataLoader,
#             BenchmarkName.MULTIARITH: MultiArithDataLoader
#         }

#         loader_class = loader_mapping.get(dataset_enum)
#         if loader_class:
#             return loader_class(os.path.join('benchmark', dataset_name, datafile_name), **kwargs)
#         return None

class BBHDataLoader(BaseDataLoader):
    data: List[Dict[str, Any]] = []
    def load_data(self) -> List[Dict[str, Any]]:
        with open(self.file_path, 'r') as file:
            json_data = json.load(file)
            self.data = json_data.get('examples', [])
        return self.data

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self.load_data()
        return iter(self.data)


class AQuADataLoader(JsonlDataLoader):
    pass


class GSMDataLoader(CsvDataLoader):
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(
            self.file_path,
            header=None,
            sep="\t",
            encoding='utf-8'
        )


class MMLUDataLoader(CsvDataLoader):
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(
            self.file_path,
            index_col=None,
            header=None,
            encoding='utf-8'
        )



class MultiArithDataLoader(BaseDataLoader):
    def load_data(self) -> List[Dict[str, Any]]:
        with open(self.file_path, 'r') as f:
            self.data = json.load(f)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self.load_data()
        for item in self.data:
            yield item

class THUNEWSDataLoader(JsonlDataLoader):
    pass

class CMMLUDataLoader(JsonlDataLoader):
    pass