# A file containing the json schema for the output of all the LLM chains
# A file containing the parser for the output of all the LLM chains
import re


def prediction_parser(response: dict) -> dict:
    """
    Parse the response from the LLM chain
    :param response: The response from the LLM chain
    :return: The parsed response
    """
    pattern = re.compile(r'Sample (\d+): (\w+)')
    matches = pattern.findall(response['text'])  ##TODO: qwen-plus should be response['text']
    predictions = [{'id': int(match[0]), 'prediction': match[1]} for match in matches]
    return {'results': predictions}

def prediction_generation_parser(response: dict) -> dict:
    """
    Parse the response from the LLM chain
    :param response: The response from the LLM chain
    :return: The parsed response
    """
    pattern = re.compile(r'Sample (\d+): (.*?)(?=<eos>|$)', re.DOTALL)
    matches = pattern.findall(response['output']['text'])
    predictions = [{'id': int(match[0]), 'prediction': match[1].strip()} for match in matches]
    return {'results': predictions}

# def prediction_chinese_parser(response: dict) -> dict:
#     """
#     Parse the response from the LLM chain
#     :param response: The response from the LLM chain
#     :return: The parsed response
#     """
#     pattern = re.compile(r'Sample (\d+): (\w+)')
#     matches = pattern.findall(response['text'])
#     predictions = [{'id': int(match[0]), 'prediction': match[1]} for match in matches]
#     return {'results': predictions}

# def prediction_generation_chinese_parser(response: dict) -> dict:
#     """
#     Parse the response from the LLM chain
#     :param response: The response from the LLM chain
#     :return: The parsed response
#     """
#     pattern = re.compile(r'Sample (\d+): (.*?)(?=<eos>|$)', re.DOTALL)
#     matches = pattern.findall(response['text'])
#     predictions = [{'id': int(match[0]), 'prediction': match[1].strip()} for match in matches]
#     return {'results': predictions}

prediction_generation_chinese_parser = prediction_generation_parser
prediction_chinese_parser = prediction_parser
