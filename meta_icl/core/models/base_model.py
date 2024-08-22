import inspect
import time
from abc import abstractmethod, ABCMeta
from typing import Any, Union, List
import asyncio
from tqdm.asyncio import tqdm
import openai

from meta_icl.core.scheme.message import Message
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
        self._call_module: Any = None
        
        self.logger = Logger.get_logger()
        
    @property
    def call_module(self):
        if self._call_module is None:
            if self.module_name not in MODEL_REGISTRY.module_dict:
                raise RuntimeError(f"method_type={self.module_name} is not supported!")
            obj_cls = MODEL_REGISTRY[self.module_name]

            if self.kwargs_filter:
                allowed_kwargs = list(inspect.signature(obj_cls.__init__).parameters.keys())
                kwargs = {key: value for key, value in self.kwargs.items() if key in allowed_kwargs}
            else:
                kwargs = self.kwargs
            self._call_module = obj_cls(**kwargs)
        return self._call_module

    @abstractmethod
    def _call(self, stream: bool = False, **kwargs) -> ModelResponse:
        """
        :param kwargs:
        :return:
        """

    def call(self, stream: bool = False, prompt: str = "", messages: List[Message] = [], **kwargs) -> ModelResponse:
        """
        :param stream: only llm needs stream
        :param kwargs:
        :return:
        """
        model_response = ModelResponse(m_type=self.m_type)
        if prompt and messages:
            raise ValueError("prompt and messages cannot be both specified")
        
        self.kwargs.update(**kwargs)
        with Timer(self.__class__.__name__, log_time=False, use_ms=False) as t:
            for i in range(self.max_retries):
                if self.raise_exception:
                    call_result = self._call(stream=stream, prompt=prompt, messages=messages, **self.kwargs)
                    if hasattr(call_result, 'status_code') and call_result.status_code != 200:
                        time.sleep(i * self.retry_interval)
                    else:
                        model_response.raw = call_result
                        break
                else:
                    try:
                        call_result = self._call(stream=stream, prompt=prompt, messages=messages, **self.kwargs)
                        if hasattr(call_result, 'status_code') and call_result.status_code != 200:
                            time.sleep(i * self.retry_interval)
                        else:
                            break
                    except (Exception, openai.OpenAIError) as e:
                        self.logger.info(f"call model={self.model_name} failed! details={e.args}, fail times={i+1}")

            if not model_response.raw:
                self.logger.warning(f"Called {self.model_name} {self.max_retries} times, max retries reached!", stacklevel=2)

        return self.after_call(stream=stream, model_response=model_response, **kwargs)
    
    @abstractmethod
    def after_call(self, model_response: ModelResponse, **kwargs) -> Union[ModelResponse, ModelResponseGen]:
        pass

class BaseAsyncModel(metaclass=ABCMeta):
    m_type: Union[ModelEnum, None] = None

    def __init__(self,
                 model_name: str,
                 module_name: str,
                 timeout: int = None,
                 max_retries: int = 3,
                 retry_interval: float = 1.0,
                 kwargs_filter: bool = True,
                 raise_exception: bool = False,
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
        self._call_module: Any = None
        
        self.logger = Logger.get_logger()
        
    @property
    def call_module(self):
        if self._call_module is None:
            if self.module_name not in MODEL_REGISTRY.module_dict:
                raise RuntimeError(f"method_type={self.module_name} is not supported!")
            obj_cls = MODEL_REGISTRY[self.module_name]

            if self.kwargs_filter:
                allowed_kwargs = list(inspect.signature(obj_cls.__init__).parameters.keys())
                kwargs = {key: value for key, value in self.kwargs.items() if key in allowed_kwargs}
            else:
                kwargs = self.kwargs
            self._call_module = obj_cls(**kwargs)
        return self._call_module

    @abstractmethod
    async def _async_call(self, **kwargs) -> Any:
        """
        :param kwargs:
        :return:
        """
    async def async_call(self, prompts: List[str] = [], list_of_messages: List[List[Message]] = [], semaphore: int = 10, **kwargs) -> dict:
        semaphore = asyncio.Semaphore(semaphore)
        self.kwargs.update(**kwargs)
        # print(self.kwargs)
        if prompts and list_of_messages:
            raise ValueError("prompt and messages cannot be both specified")
        async def task(index, prompt: str = "", messages: List[Message] = [], **kwargs):
            
            async with semaphore:
            # async with self.limiter:
                model_response = ModelResponse(m_type=self.m_type)
                for i in range(self.max_retries):
                    if self.raise_exception:
                        call_output = await self._async_call(prompt=prompt, messages = messages, **self.kwargs)
                        if hasattr(call_output, 'status_code') and call_output.status_code != 200:
                            self.logger.info(f"async_call model={self.model_name} failed! index={index}, details={e.args}, fail times={i+1}")
                            asyncio.sleep(self.retry_interval)
                        else:
                            model_response.raw = call_output
                            break
                    else:
                        try:
                            call_output = await self._async_call(prompt=prompt, messages = messages, **self.kwargs)
                            if hasattr(call_output, 'status_code') and call_output.status_code != 200:
                                self.logger.info(f"async_call model={self.model_name} failed! index={index}, details={e.args}, fail times={i+1}")
                                asyncio.sleep(self.retry_interval)
                            else:
                                model_response.raw = call_output
                                break
                        except (Exception, openai.OpenAIError) as e:
                            self.logger.info(f"async_call model={self.model_name} failed! index={index}, details={e.args}, fail times={i+1}")
                            await asyncio.sleep(self.retry_interval)

                if not model_response.raw:
                    self.logger.warning(f"Called {self.model_name} {self.max_retries} times, max retries reached!", stacklevel=2)

                return {"response": self.after_call(model_response=model_response, **kwargs), "index": index}

        if prompts:
            tasks = [task(i, prompt=prompts[i], **kwargs) for i in range(len(prompts))]
        elif list_of_messages:
            tasks = [task(i, messages=list_of_messages[i], **kwargs) for i in range(len(list_of_messages))]
        else:
            return

        
        # for _ in model_responses:
        #     pbar.update(1)
        model_responses = []
        for result in tqdm.as_completed(tasks):
            # As each task completes, the progress bar will be updated
            value = await result
            model_responses.append(value)
    # return await asyncio.gather(*responses)
        # pbar.close()
        model_responses = sorted(model_responses, key=lambda x: x["index"])

        # print(t.cost_str)
        return [model_response["response"] for model_response in model_responses]