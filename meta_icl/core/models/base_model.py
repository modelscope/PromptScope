import inspect
import time
from abc import abstractmethod, ABCMeta
from typing import Any, Union

from meta_icl.core.enumeration.model_enum import ModelEnum
from meta_icl.core.utils.registry import Registry
from meta_icl.core.scheme.model_response import ModelResponse, ModelResponseGen
from meta_icl.core.utils.timer import Timer
from meta_icl.core.utils.logger import Logger

MODEL_REGISTRY = Registry("models")


class BaseModel(metaclass=ABCMeta):
    m_type: Union[ModelEnum, None] = None

    def __init__(self,
                 model_name: str,
                 module_name: str,
                 timeout: int = None,
                 max_retries: int = 3,
                 retry_interval: float = 1.0,
                 kwargs_filter: bool = True,
                 raise_exception: bool = True,
                 **kwargs):

        self.model_name: str = model_name
        self.module_name: str = module_name
        self.timeout: int = timeout
        self.max_retries: int = max_retries
        self.retry_interval: float = retry_interval
        self.kwargs_filter: bool = kwargs_filter
        self.raise_exception: bool = raise_exception
        self.kwargs: dict = kwargs

        self.data = {}
        self._model: Any = None
        
        self.logger = Logger.get_logger()
        
    @property
    def model(self):
        if self._model is None:
            if self.module_name not in MODEL_REGISTRY.module_dict:
                raise RuntimeError(f"method_type={self.module_name} is not supported!")
            obj_cls = MODEL_REGISTRY[self.module_name]

            if self.kwargs_filter:
                allowed_kwargs = list(inspect.signature(obj_cls.__init__).parameters.keys())
                kwargs = {key: value for key, value in self.kwargs.items() if key in allowed_kwargs}
            else:
                kwargs = self.kwargs
            self._model = obj_cls(**kwargs)
        return self._model

    @abstractmethod
    def before_call(self, **kwargs) -> None:
        """prepare data before call
        :param kwargs:
        :return:
        """

    @abstractmethod
    def after_call(self, model_response: Union[ModelResponse, ModelResponseGen],
                   **kwargs) -> Union[ModelResponse, ModelResponseGen]:
        """
        :param model_response:
        :param kwargs:
        :return:
        """

    @abstractmethod
    def _call(self, stream: bool = False, **kwargs) -> Union[ModelResponse, ModelResponseGen]:
        """
        :param kwargs:
        :return:
        """

    def call(self, stream: bool = False, **kwargs) -> Union[ModelResponse, ModelResponseGen]:
        """
        :param stream: only llm needs stream
        :param kwargs:
        :return:
        """
        with Timer(self.__class__.__name__, log_time=False) as t:
            self.before_call(stream=stream, **kwargs)
            for i in range(self.max_retries):
                if self.raise_exception:
                    model_response = self._call(stream=stream, **kwargs)
                else:
                    try:
                        model_response = self._call(stream=stream, **kwargs)
                    except Exception as e:
                        model_response = ModelResponse(m_type=self.m_type, status=False, details=e.args)

                if isinstance(model_response, ModelResponse) and not model_response.status:
                    self.logger.warning(f"call model={self.model_name} failed! cost={t.cost_str} retry_cnt={i} "
                                        f"details={model_response.details}", stacklevel=2)
                    time.sleep(i * self.retry_interval)
                else:
                    return self.after_call(stream=stream, model_response=model_response, **kwargs)

    @abstractmethod
    async def _async_call(self, **kwargs) -> ModelResponse:
        """
        :param kwargs:
        :return:
        """

    async def async_call(self, **kwargs) -> ModelResponse:
        """ 异步不需要stream
        :param kwargs:
        :return:
        """
        with Timer(self.__class__.__name__, log_time=False) as t:
            self.before_call(**kwargs)
            for i in range(self.max_retries):
                if self.raise_exception:
                    model_response = self._async_call(**kwargs)
                else:
                    try:
                        model_response = self._async_call(**kwargs)
                    except Exception as e:
                        model_response = ModelResponse(m_type=self.m_type, status=False, details=e.args)

                if not model_response.status:
                    self.logger.warning(f"async_call model={self.model_name} failed! cost={t.cost_str} retry_cnt={i} "
                                        f"details={model_response.details}", stacklevel=2)
                    time.sleep(i * self.retry_interval)
                else:
                    return self.after_call(model_response=model_response, **kwargs)
