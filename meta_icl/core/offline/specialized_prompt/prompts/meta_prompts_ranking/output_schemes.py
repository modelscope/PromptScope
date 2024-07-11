# A file containing the json schema for the output of all the LLM chains
import json
import re

initial_schema = step_samples_schema = {
  "description": "A List of all results",
  "properties": {
    "samples": {
      "description": "Each sample is a string containing the sample content, without any additional information like the Prediction or GT",
      "items": {
        "type": "string"
      },
      "title": "Samples",
      "type": "array"
    }
  },
  "required": [
    "samples"
  ],
  "title": "Sample_List",
  "type": "object"
}


classification_prediction_schema = {
  "$defs": {
    "Result": {
      "description": "A single result",
      "properties": {
        "id": {
          "description": "The sample id",
          "title": "Id",
          "type": "integer"
        },
        "prediction": {
          "description": "The prediction of the sample.",
          "title": "Prediction",
          "type": "string"
        }
      },
      "required": [
        "id",
        "prediction"
      ],
      "title": "Result",
      "type": "object"
    }
  },
  "description": "A List of task classification results",
  "properties": {
    "results": {
      "description": "Each item contain the id and the prediction of the sample",
      "items": {
        "$ref": "#/$defs/Result"
      },
      "title": "Results",
      "type": "array"
    }
  },
  "required": [
    "results"
  ],
  "title": "Results_List",
  "type": "object"
}


step_prompt_schema = {
  "description": "A prompt suggestion which expect to get high score, and the associated score prediction",
  "properties": {
    "prompt": {
      "description": "The prompt prediction",
      "title": "Prompt",
      "type": "string"
    },
    "score": {
      "description": "The score prediction",
      "title": "Score",
      "type": "number"
    }
  },
  "required": [
    "prompt",
    "score"
  ],
  "title": "Suggested_Prompt",
  "type": "object"
}

def update_classification_prediction_schema(label_schema:list)->dict:
  """
  Updates the classification prediction schema with the label schema from the yaml file
  :param yaml_data: The yaml data
  """

  classification_prediction_schema['$defs']['Result']['properties']['prediction']['enum'] = label_schema
  classification_prediction_schema['$defs']['Result']['properties']['prediction'][
    'description'] += 'The answer must be one of the following options: {} !!'.format(label_schema)
  return classification_prediction_schema

def initial_parser(result):
  samples = result['text'].split("\n\n")
  # output = ["\n\n".join(samples[i:i+2]) for i in range(0, len(samples)-1, 2)]
  output = [x for x in samples if x.startswith('Sample')]
  print('ranking output after parser is:', output)
  return {'samples': output}

step_samples_parser = initial_parser

def step_prompt_parser(result):
  text = result['text']
  # prompt = prompt.split("\n")
  # prompt_dict = {}
  # prompt_dict['prompt'] = prompt[0].split(':')[-1]
  # prompt_dict['score'] = prompt[1].split(':')[-1]
  delimiters = ['prompt:', 'score:']
  # pattern = '|'.join([f'\\b{delimiter}\\b' for delimiter in delimiters])
  pattern = '|'.join([f'{delimiter}' for delimiter in delimiters])
  compiled_pattern = re.compile(pattern, re.IGNORECASE)
  split_text = compiled_pattern.split(text)
  split_text = [x for x in split_text if x]
  prompt_dict = {}
  prompt_dict['prompt'] = ''.join(split_text[:-1])
  prompt_dict['score'] = split_text[-1]

  return prompt_dict

def initial_chinese_parser(result):
  samples = result['text'].split("\n\n")
  output = [x for x in samples if x.startswith('样本')]
  print('ranking output after parser is:', output)
  return {'samples': output}

step_samples_chinese_parser = initial_chinese_parser

def step_prompt_chinese_parser(result):
  prompt = result['text']
  prompt = prompt.split("\n")
  prompt_dict = {}
  prompt_dict['prompt'] = prompt[0].split(':')[-1]
  prompt_dict['score'] = prompt[1].split(':')[-1]
  return prompt_dict