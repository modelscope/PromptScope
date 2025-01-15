import os
from typing import Any, List, Optional, Union
from enum import Enum
from pydantic import PrivateAttr
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, InternalServerError
from llamakit.core.schemas.types import Embedding
from llamakit.core.utils.logger import Logger
from llamakit.core.embeddings.base import BaseEmbedding
from llamakit.core.utils.iterator import batched
from llamakit.core.utils.wrappers import create_retry_decorator

logger = Logger.get_logger()
OPENAI_MAX_BATCH_SIZE = 2048
OPENAI_MAX_INPUT = 8191
DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"

embedding_retry = create_retry_decorator(
    max_retries=3,
    exceptions=(
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        InternalServerError,
    ),
    random_exponential=True,
    stop_after_delay_seconds=60,
    min_seconds=1,
    max_seconds=20,
)


class OpenAIEmbeddingModelType(str, Enum):
    """OpenAI embedding model type."""

    TEXT_EMBED_ADA_002 = "text-embedding-ada-002"
    TEXT_EMBED_3_LARGE = "text-embedding-3-large"
    TEXT_EMBED_3_SMALL = "text-embedding-3-small"


DEFAULT_DIMENSIONS = {
    OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002: 1536,
    OpenAIEmbeddingModelType.TEXT_EMBED_3_SMALL: 1536,
    OpenAIEmbeddingModelType.TEXT_EMBED_3_LARGE: 3072
}


def create_openai_client(api_key: str, base_url: str = None) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url or DEFAULT_OPENAI_API_BASE)


@embedding_retry
def _openai_embedding_call(
    client: OpenAI,
    model: str,
    texts: Union[str, List[str]],
    **kwargs: Any
) -> List[Embedding]:
    if isinstance(texts, str):
        texts = [texts]
    assert len(texts) <= OPENAI_MAX_BATCH_SIZE, \
        f"The single time request should not be larger than {OPENAI_MAX_BATCH_SIZE}."

    list_of_text = [text.replace("\n", " ") for text in texts]
    data = client.embeddings.create(input=list_of_text, model=model, **kwargs).data
    return [d.embedding for d in data]


def get_text_embeddings(
    client: OpenAI,
    model: str,
    texts: List[str],
    batch_size: int = OPENAI_MAX_BATCH_SIZE,
    **kwargs: Any
) -> List[Embedding]:
    result = []  # merge the results.
    for batch in batched(texts, batch_size):
        try:
            embedding_results = _openai_embedding_call(
                client=client,
                model=model,
                texts=texts,
                **kwargs
            )
            result.extend(embedding_results)
        except Exception as e:
            logger.error(f"Failed to get embedding for text: {batch}, message {e}")
            result.extend([None] * len(batch))
    return result


class OpenAIEmbedding(BaseEmbedding):

    _api_key: Optional[str] = PrivateAttr(default=None)
    _base_url: Optional[str] = PrivateAttr(default=None)
    _client: Optional[OpenAI] = PrivateAttr(default=None)

    def __init__(
            self,
            model: OpenAIEmbeddingModelType = "text-embedding-ada-002",
            dimension: Optional[int] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            batch_size: int = OPENAI_MAX_BATCH_SIZE,
            **kwargs: Any,
    ):
        super().__init__(
            model=model,
            batch_size=batch_size,
            dimension=dimension or DEFAULT_DIMENSIONS[model]
        )
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url, **kwargs)

    def get_query_embedding(self, query: str) -> Embedding:
        return self.get_text_embedding(query)

    def get_text_embedding(self, text: str) -> Embedding:
        try:
            emb = _openai_embedding_call(
                client=self._client,
                model=self.model,
                texts=text,
                dimensions=self.dimension
            )
        except Exception as e:
            logger.error(f"Failed to get embedding for text: {text}, message {e}")
        else:
            if len(emb) > 0:
                return emb[0]
            else:
                return []

    def get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        return get_text_embeddings(
            client=self._client,
            model=self.model,
            texts=texts,
            dimensions=self.dimension,
            batch_size=self.batch_size
        )
