import os
from http import HTTPStatus
from typing import Any, List, Optional, Union, Literal, Dict
from enum import Enum
from requests.exceptions import HTTPError
from pydantic import BaseModel, Field, PrivateAttr
from prompt_scope.core.schemas.types import Embedding
from prompt_scope.core.utils.logger import Logger
from prompt_scope.core.embeddings.base import BaseEmbedding
from prompt_scope.core.utils.wrappers import retry
from prompt_scope.core.utils.iterator import batched
from prompt_scope.core.utils.similarities import cosine_similarity, lexical_similarity

logger = Logger.get_logger()
DASHSCOPE_MAX_BATCH_SIZE = 20


class EmbeddingOutputType(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    DENSE_AND_SPARSE = "dense&sparse"


class DashScopeEmbeddingName(str, Enum):
    TEXT_EMBEDDING_V1 = "text-embedding-v1"
    TEXT_EMBEDDING_V2 = "text-embedding-v2"
    TEXT_EMBEDDING_V3 = "text-embedding-v3"


class DashScopeSparseEmbedding(BaseModel):
    """Sparse embedding."""

    index: int
    value: float
    token: str


class DashScopeEmbeddingResult(BaseModel):
    text_index: int
    embedding: Optional[Embedding] = None
    sparse_embedding: Optional[List[DashScopeSparseEmbedding]] = None


class DashScopeEmbeddingOutput(BaseModel):
    embeddings: List[DashScopeEmbeddingResult] = Field(
        default_factory=list, description="Dashscope embedding output."
    )


def get_lexical_dict(sparse_embedding: List[DashScopeSparseEmbedding]) -> Dict:
    lexical_dict = {}
    for item in sparse_embedding:
        if v := lexical_dict.get(item.token):
            lexical_dict[item.token] = max(v, item.value)
        else:
            lexical_dict[item.token] = item.value
    return lexical_dict


@retry(max_retries=3, wait_time=3)
def _dashscope_embedding_call(
        model: str,
        text: Union[str, List[str]],
        dimension: int = 1024,
        api_key: Optional[str] = None,
        text_type: str = "document",
        output_type: str = EmbeddingOutputType.DENSE,
        **kwargs: Any,
) -> List[Union[Embedding, List[DashScopeSparseEmbedding], DashScopeEmbeddingResult]]:
    """Call DashScope text embedding.
       ref: https://help.aliyun.com/zh/model-studio/developer-reference/general-text-embedding.

    Args:
        model (str): The `DashScopeTextEmbeddingModels`
        text (Union[str, List[str]]): text or list text to embedding.
        dimension (int): Embedding dimension.
        api_key: Dashscope api key.
        text_type: query or document.
        output_type: dense, sparse, dense&sparse.
    Raises:
        ImportError: need import dashscope

    Returns:
        List[Embedding]: The list of embedding result, if failed return empty list.
            if some input text have no output, the corresponding index of output is None.
    """
    try:
        import dashscope
    except ImportError:
        raise ImportError("DashScope requires `pip install dashscope")

    response = dashscope.TextEmbedding.call(
        model=model, input=text, dimension=dimension, api_key=api_key,
        text_type=text_type, output_type=output_type,
        kwargs=kwargs
    )

    if response.status_code == HTTPStatus.OK:
        embedding_output = DashScopeEmbeddingOutput(**response.output)
        embedding_results = sorted(embedding_output.embeddings, key=lambda x: x.text_index)
        if output_type == EmbeddingOutputType.DENSE:
            return [rs.embedding for rs in embedding_results]
        elif output_type == EmbeddingOutputType.SPARSE:
            return [rs.sparse_embedding for rs in embedding_results]
        else:
            return embedding_results
    elif response.status_code in [400, 401]:
        logger.error(f"Failed to get embedding for text: {text}, message {response.message}")
        raise ValueError(
            f"status_code: {response.status_code} \n "
            f"code: {response.code} \n message: {response.message}"
        )
    else:
        raise HTTPError(
            f"HTTP error occurred: status_code: {response.status_code} \n "
            f"code: {response.code} \n message: {response.message}",
            response=response
        )


def get_text_embeddings(
        model: str,
        texts: List[str],
        dimension: int = 1024,
        api_key: Optional[str] = None,
        text_type: str = "document",
        output_type: str = EmbeddingOutputType.DENSE,
        batch_size: int = DASHSCOPE_MAX_BATCH_SIZE,
        **kwargs: Any,
) -> List[Union[Embedding, List[DashScopeSparseEmbedding], DashScopeEmbeddingResult]]:
    """Call DashScope text embedding.
       ref: https://help.aliyun.com/zh/model-studio/developer-reference/general-text-embedding.

    Args:
        model (str): The `DashScopeTextEmbeddingModels`
        dimension (int): The `DashScopeTextEmbeddingModels dimension`
        texts (List[str]): text or list text to embedding.
        api_key: Dashscope api key.
        text_type: query or document.
        batch_size: Batch size for request.
        output_type: dense, sparse, dense&sparse.
    Raises:
        ImportError: need import dashscope

    Returns:
        List[Embedding]: The list of embedding result, if failed return empty list.
            if some input text have no output, the corresponding index of output is None.
    """
    result = []  # merge the results.
    for batch in batched(texts, batch_size):
        try:
            embedding_results = _dashscope_embedding_call(
                model, batch, dimension=dimension, api_key=api_key, text_type=text_type,
                output_type=output_type,
                kwargs=kwargs
            )
            result.extend(embedding_results)
        except Exception as e:
            logger.error(f"Failed to get embedding for text: {batch}, message {e}")
            result.extend([None] * len(batch))

    return result


class DashScopeEmbedding(BaseEmbedding):

    _api_key: Optional[str] = PrivateAttr(default=None)

    def __init__(
            self,
            model: str = "text-embedding-v3",
            dimension: int = 1024,
            api_key: Optional[str] = None,
            batch_size: int = DASHSCOPE_MAX_BATCH_SIZE,
            **kwargs: Any,
    ):
        super().__init__(
            model=model,
            batch_size=batch_size,
            dimension=dimension,
            **kwargs
        )
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")

    def get_query_embedding(self, query: str) -> Embedding:
        try:
            emb = _dashscope_embedding_call(
                self.model,
                query,
                dimension=self.dimension,
                api_key=self._api_key,
                text_type="query",
                output_type=EmbeddingOutputType.DENSE
            )
        except Exception as e:
            logger.error(f"Failed to get embedding for text: {query}, message {e}")
        else:
            if len(emb) > 0:
                return emb[0]
            else:
                return []

    def get_text_embedding(self, text: str) -> Embedding:
        try:
            emb = _dashscope_embedding_call(
                self.model,
                text,
                dimension=self.dimension,
                api_key=self._api_key,
                text_type="document",
                output_type=EmbeddingOutputType.DENSE
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
            self.model,
            texts,
            dimension=self.dimension,
            batch_size=self.batch_size,
            api_key=self._api_key,
            text_type="document",
            output_type=EmbeddingOutputType.DENSE
        )

    def get_sparse_embedding(self, text: str) -> List[DashScopeSparseEmbedding]:
        if self.model != DashScopeEmbeddingName.TEXT_EMBEDDING_V3:
            raise ValueError(f"Only {DashScopeEmbeddingName.TEXT_EMBEDDING_V3} support sparse embedding.")

        try:
            emb = _dashscope_embedding_call(
                self.model,
                text,
                api_key=self._api_key,
                output_type=EmbeddingOutputType.SPARSE
            )
        except Exception as e:
            logger.error(f"Failed to get embedding for text: {text}, message {e}")
        else:
            if len(emb) > 0:
                return emb[0]
            else:
                return []

    def get_sparse_embeddings(self, texts: List[str]) -> List[List[DashScopeSparseEmbedding]]:
        return get_text_embeddings(
            self.model,
            texts,
            dimension=self.dimension,
            batch_size=self.batch_size,
            api_key=self._api_key,
            output_type=EmbeddingOutputType.SPARSE
        )

    def get_text_similarities(
            self,
            query: str,
            texts: List[str],
            embedding_type: Literal["dense", "sparse"] = "dense"
    ) -> List[float]:
        if embedding_type == EmbeddingOutputType.DENSE:
            return list(cosine_similarity(
                [self.get_query_embedding(query)],
                self.get_text_embeddings(texts),
                squeeze=True
            ))
        else:
            query_embedding = get_lexical_dict(self.get_sparse_embedding(query))
            document_embeddings = self.get_sparse_embeddings(texts)
            document_embeddings = [get_lexical_dict(de) for de in document_embeddings]
            return [lexical_similarity(query_embedding, de) for de in document_embeddings]


