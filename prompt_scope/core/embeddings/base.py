"""Base embedding interface."""

from abc import abstractmethod
from typing import List, Any
from pydantic import BaseModel, Field
from prompt_scope.core.schemas.types import Embedding
from prompt_scope.core.bases.mixin import NodeListTransformMixin
from prompt_scope.core.schemas.node import BaseNode, ContentMode


class BaseEmbedding(BaseModel, NodeListTransformMixin[BaseNode]):

    model: str
    batch_size: int = Field(default=16, description="Embedding call batch size.")
    dimension: int = Field(default=1024, description="Embedding model dimension.")

    @abstractmethod
    def get_query_embedding(self, query: str) -> Embedding:
        """Embedded the input query.
        Args:
            query:

        Returns:

        """
        raise NotImplementedError()

    def aget_query_embedding(self, query: str) -> Embedding:
        """Asynchronously embedded the input query.
        Args:
            query:

        Returns:

        """

    @abstractmethod
    def get_text_embedding(self, text: str) -> Embedding:
        """Embedded the input text.
        Args:
            text:

        Returns:
        """

    def aget_text_embedding(self, text: str) -> Embedding:
        """Embedded the input text.
        Args:
            text:

        Returns:
        """

    @abstractmethod
    def get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """Embedded the input text.
        Args:
            texts:

        Returns:
        """

    def aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """Embedded the input text.
        Args:
            texts:

        Returns:
        """

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        contents = [node.get_content(content_mode=ContentMode.EMBED) for node in nodes]
        embedding = self.get_text_embeddings(contents)
        for node, emb in zip(nodes, embedding):
            node.embedding = emb

        return nodes
