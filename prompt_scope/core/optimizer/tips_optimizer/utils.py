import numpy as np
from tqdm.asyncio import tqdm
from typing import List, Callable, Tuple
import asyncio
import nest_asyncio
nest_asyncio.apply()

from prompt_scope.core.schemas.example import LLMCallRecord, ConversationRound
from prompt_scope.core.llms.dashscope_llm import DashscopeLLM
from prompt_scope.core.schemas.message import ChatMessage, MessageRole



async def async_predict_on_dataset(llm, list_of_messages, semaphore, show_process_bar):
    tasks = [async_predict_on_sample(llm, i, messages, semaphore) for i, messages in enumerate(list_of_messages)]
    responses = []

    if show_process_bar:
        for result in tqdm.as_completed(tasks, ncols=80, desc="Predict"):
            value = await result
            responses.append(value)
    else:
        for result in tasks:
            value = await result
            responses.append(value)
    return [response["result"] for response in sorted(responses, key=lambda x: x["index"])]


async def async_predict_on_sample(infer_llm: DashscopeLLM, index, messages, semaphore):
    async with semaphore:
        result = await asyncio.to_thread(predict_on_sample, infer_llm, messages)
    return {"index": index, "result": result}


def predict_on_sample(infer_llm: DashscopeLLM, messages):
    return infer_llm.chat(messages=messages).message.content


def evaluate_on_sample(infer_llm: DashscopeLLM, sample: LLMCallRecord, is_good_case_func: Callable[[str, str], Tuple]):
    messages = transfer_sample_to_messages(sample)
    # print(messages)
    prediction = predict_on_sample(infer_llm, messages)
    # print(prediction)
    sample.prediction = prediction
    is_good_case, score = is_good_case_func(prediction, sample.output)
    sample.is_good_case = is_good_case
    sample.score = score
    return score


def transfer_sample_to_messages(sample):
    messages = []
    system_prompt = sample.system_prompt
    if system_prompt is not None:
        if sample.tips is not None and len(sample.tips) > 0:
            system_prompt = f"{system_prompt}\n\n{sample.tips}"
        messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))
    if sample.history is not None:
        for conversation_round in sample.history:
            messages.append(ChatMessage(role=MessageRole.USER, content=conversation_round.user))
            messages.append(ChatMessage(role=MessageRole.USER, content=conversation_round.assistant))

    messages.append(ChatMessage(role=MessageRole.USER, content=sample.input))
    return messages


def predict_on_dataset(infer_llm: DashscopeLLM, dataset: List[LLMCallRecord], tips, semaphore, show_process_bar):
    dataset_copy = [d.model_copy(deep=True) for d in dataset]

    list_of_messages = []
    for idx, sample in enumerate(dataset_copy):
        sample.tips = tips
        message = transfer_sample_to_messages(sample)
        list_of_messages.append(message)

    prediction_list = asyncio.run(async_predict_on_dataset(llm=infer_llm, list_of_messages=list_of_messages, semaphore=semaphore, show_process_bar=show_process_bar))

    dataset_with_prediction = []
    for idx, sample in enumerate(dataset_copy):
        sample.prediction = prediction_list[idx]
        dataset_with_prediction.append(sample)

    return dataset_with_prediction


def evaluate_on_dataset(infer_llm: DashscopeLLM, dataset: List[LLMCallRecord],
                        is_good_case_func: Callable[[str, str], Tuple], tips=None, semaphore=None, show_process_bar=True):
    dataset_with_prediction = predict_on_dataset(infer_llm, dataset, tips, semaphore, show_process_bar)
    score_results = score_on_dataset(dataset_with_prediction, is_good_case_func, semaphore, show_process_bar)
    dataset_with_score = []
    score_list = []

    for idx, sample in enumerate(dataset_with_prediction):
        is_good_case, score = score_results[idx]
        score_list.append(score)
        sample.is_good_case = is_good_case
        sample.score = score
        dataset_with_score.append(sample)

    return np.mean(score_list), dataset_with_score


def score_on_dataset(dataset_with_prediction: List[LLMCallRecord], is_good_case_func: Callable[[str, str], Tuple], semaphore, show_process_bar):
    score_result_list = asyncio.run(async_score_on_dataset(dataset_with_prediction, is_good_case_func, semaphore, show_process_bar))
    return score_result_list


async def async_score_on_dataset(dataset_with_prediction: List[LLMCallRecord], is_good_case_func: Callable[[str, str], Tuple], semaphore, show_process_bar):
    tasks = [async_is_good_case(i, sample, is_good_case_func, semaphore) for i, sample in enumerate(dataset_with_prediction)]
    responses = []

    if show_process_bar:
        for result in tqdm.as_completed(tasks, ncols=80, desc="Score"):
            value = await result
            responses.append(value)
    else:
        for result in tasks:
            value = await result
            responses.append(value)
    return [response["result"] for response in sorted(responses, key=lambda x: x["index"])]


async def async_is_good_case(index, sample, is_good_case_func, semaphore):
    async with semaphore:
        result = await asyncio.to_thread(is_good_case_func, sample.prediction, sample.output)
        return {"index": index, "result": result}