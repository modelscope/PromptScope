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
  # output = []
  samples = result['text']
  # tmps = samples.split("Answer: Yes")
  # # print(tmps)
  # for tmp in tmps:
  #   output += tmp.split("Answer: No")
  #   # print(output)
  # output = [x.strip() for x in output if len(x) > 0]
  # print(output)
  output = [x.strip() for x in samples.split("\n") if len(x) > 0 and bool(re.match(r'^\d', x))]
  print('output after parser is:', output)
  return {'samples': output}

step_samples_parser = initial_parser

def step_prompt_parser(result):
  prompt = result['text']
  prompt = prompt.replace("\n", "")
  return json.loads(prompt)

def initial_chinese_parser(result):
  output = result['text'].split("\n")[1:]
  return {'samples': output}

step_samples_chinese_parser = initial_chinese_parser

def step_prompt_chinese_parser(result):
  prompt = result['text']
  prompt = prompt.split("\n")
  prompt_dict = {}
  prompt_dict['prompt'] = prompt[0].split(':')[-1]
  prompt_dict['score'] = prompt[1].split(':')[-1]
  return prompt_dict
# Sample of Tongyi Output: [{'num_samples': 10, 'task_description': 'Assistant is an expert classifier that will classify a movie review, and let the user know if it contains a spoiler for the reviewed movie or not.', 'instruction': 'Does this movie review contain a spoiler? answer Yes or No', 'text': '1. Review: "The film\'s climax, where the protagonist discovers the villain\'s true identity, left me gasping." Answer: Yes\n2. Review: "I loved how the director kept the twist ending a secret until the very end." Answer: Yes\n3. Review: "The scene where they find the treasure in the hidden room was so unexpected and thrilling." Answer: Yes\n4. Review: "The opening sequence with the spaceship crash sets the tone for the entire movie." Answer: No\n5. Review: "The acting was superb, especially considering the complex characters\' backstories." Answer: No\n6. Review: "The movie\'s soundtrack added depth to the emotional journey of the main character." Answer: No\n7. Review: "Without giving too much away, the resolution to the time-travel paradox was brilliant." Answer: Yes\n8. Review: "I was disappointed by the lack of development in the romantic subplot." Answer: No\n9. Review: "The cinematography captured the essence of the exotic locations beautifully." Answer: No\n10. Review: "The final fight scene, where the hero defeats the enemy, is worth the price of admission alone." Answer: Yes'}]