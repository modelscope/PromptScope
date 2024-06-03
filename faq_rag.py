"""
LlamaIndex for FAQ support.

Package requirements:
pip install llama-index-core
pip install llama-index-postprocessor-dashscope-rerank-custom

TODO:
1. add sparse index and retriever support.
2. update the ranking algorithm in FAQ.
3. add category info to embedding.
"""

import os
import pandas as pd
import time
import logging
from functools import wraps
from typing import Union, List, Tuple, Optional
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from llama_index.core.postprocessor import SimilarityPostprocessor
from tqdm import tqdm


Settings.embed_model = DashScopeEmbedding(model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2)
MAX_SIMILAR_QUESTIONS = 15
SIMILARITY_TOP_K = 10
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Function {func.__name__!r} execute with {duration:.4f} s")
        return result
    return wrapper


class FaqHandler(object):

    def __init__(
            self,
            file_path: str,
            question_column: str,
            similar_columns: Union[str, List],
            answer_column: str,
            similar_seperator: str = "\n",
            embedding_similar_columns: bool = True,
            persist_dir: Optional[str] = None,
            enable_rerank: bool = False,
            str_columns: Union[str, List] = None,
            similarity_cutoff: Optional[float] = None
    ):
        file_extension = os.path.splitext(file_path)[1]
        if file_extension not in (".xlsx", ".csv"):
            raise ValueError("Not supported file path for faq.")

        dtype = None
        if str_columns is not None:
            if isinstance(str_columns, str):
                str_columns = [str_columns]
            dtype = dict((col, str) for col in str_columns)

        if file_extension == '.xlsx':
            self.data = pd.read_excel(file_path, dtype=dtype)
        else:
            self.data = pd.read_csv(file_path, dtype=dtype)

        assert question_column in self.data.columns, f"Invalid input question column {question_column}."
        if isinstance(similar_columns, str):
            assert similar_columns in self.data.columns, f"Invalid input similar question column {similar_columns}."
        else:
            for sc in similar_columns:
                assert sc in self.data.columns, f"Invalid input similar questions column {sc}."
        assert answer_column in self.data.columns, f"Invalid input question column {answer_column}."

        self.question_column = question_column
        self.similar_columns = similar_columns
        self.answer_column = answer_column
        self.similar_seperator = similar_seperator
        self.embedding_similar_columns = embedding_similar_columns
        self.persist_dir = persist_dir
        self.enable_rerank = enable_rerank
        self.similarity_cutoff = similarity_cutoff
        self._index = None

    @timer
    def from_documents(self):
        if self.embedding_similar_columns:
            if isinstance(self.similar_columns, str):
                similar_questions = self.data[self.similar_columns].apply(
                    lambda value: str(value).split(self.similar_seperator)[:MAX_SIMILAR_QUESTIONS]
                    if not pd.isna(value) else []
                )
            else:
                similar_questions = self.data[self.similar_columns].apply(
                    lambda value: list(value)[:MAX_SIMILAR_QUESTIONS], axis=1
                )

            for name, sm_questions in similar_questions.items():
                question = self.data[self.question_column].loc[name]
                if not pd.isna(question):
                    sm_questions.insert(0, question)

            questions = similar_questions
        else:
            questions = self.data[self.question_column].apply(lambda value: [value])

        embedding_model = DashScopeEmbedding()
        nodes = []

        for name, row in tqdm(self.data.iterrows(), total=len(self.data)):
            question_list = questions.loc[name]
            embedding = embedding_model.get_agg_embedding_from_queries(question_list)
            nodes.append(TextNode(
                text="\n".join(question_list),
                embedding=embedding,
                metadata=row.to_dict()
            ))

        self._index = VectorStoreIndex(nodes)
        if self.persist_dir is not None:
            self._index.storage_context.persist(self.persist_dir)

    @timer
    def load_index(self):
        storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
        self._index = load_index_from_storage(storage_context)

    def retrieve(self, query: str) -> List[NodeWithScore]:
        retriever = self._index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)
        nodes = retriever.retrieve(query)

        if self.enable_rerank:
            ranker = DashScopeRerank()
            nodes = ranker.postprocess_nodes(nodes=nodes, query_str=query)

        if self.similarity_cutoff:
            filter_processor = SimilarityPostprocessor()
            filter_processor.similarity_cutoff = self.similarity_cutoff
            nodes = filter_processor.postprocess_nodes(nodes)
        return nodes

    @timer
    def query(self, query: str) -> Tuple[Optional[str], bool, float]:
        nodes = self.retrieve(query)
        if not nodes:
            return None, False, 0
        if nodes:
            node = nodes[0]
            logging.info(f"Retrieve node with similar score is {node.score: .4f}")
            return node.metadata.get("answer", ""), True, node.score


if __name__ == "__main__":
    handler = FaqHandler(
        file_path="./data/faq_data.xlsx",
        question_column="question",
        similar_columns="similar_questions",
        answer_column="answer",
        persist_dir="./data/vector_store",
        str_columns=["create_time", "update_time"],
        similarity_cutoff=0.5
    )
    """ Step 1: create index"""
    # handler.from_documents()

    # """Step 2: online query"""
    test_query = "请帮我转人工客服？"
    handler.load_index()
    answer, is_successful, sim_score = handler.query(test_query)
    if is_successful:
        logging.info(f"Query {test_query}, answer: {answer}")
    else:
        logging.info("No relevant nodes retrieved.")
