import pytest

from prompt_scope.core.evals.loading import load_evaluator
from prompt_scope.core.evals.schema import EvaluatorType


VALIDJSON = """
{

    "name": "Alice Smith",

    "age": 28,

    "city": "New York",

    "isStudent": false,

    "hobbies": ["painting", "yoga"],

    "contact": {

        "email": "alice@example.com",

        "phone": "555-1234"

    }
}
"""

INVALIDJSON = """
{

    "name": "John Smith",

    "age": 30,

    "city": "San Francisco"

    "isEmployed": true,

    "hobbies": ["reading" "hiking"],

    "contact": {

        "email": "john@example.com",

        "phone": 5551234

    },

}
"""

VALIDJSON_2 = """
{

    "book": {

        "title": "The Great Gatsby",

        "author": "F. Scott Fitzgerald",

        "year": 1925,

        "genres": ["Novel", "Fiction", "Classic"],

        "ratings": {

            "goodreads": 3.93,

            "amazon": 4.5

        },

    "inPrint": true

}

}
"""

JSONSCHEMA = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "name": {
      "type": "string"
    },
    "age": {
      "type": "integer",
      "minimum": 0
    },
    "city": {
      "type": "string"
    },
    "isStudent": {
      "type": "boolean"
    },
    "hobbies": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "contact": {
      "type": "object",
      "properties": {
        "email": {
          "type": "string",
          "format": "email"
        },
        "phone": {
          "type": "string",
          "pattern": "^[0-9-]+$"
        }
      },
      "required": ["email", "phone"]
    }
  },
  "required": ["name", "age", "city", "isStudent", "hobbies", "contact"],
  "additionalProperties": False
}

@pytest.mark.parametrize(
    "prediction, score", 
    [
        (VALIDJSON, 1.0),
        (INVALIDJSON, 0.0)
    ]
)
def test_json_validity(prediction, score) -> None:
    json_validity_evaluator = load_evaluator(
        evaluator=EvaluatorType.JSON_VALIDITY
    )
    result = json_validity_evaluator.evaluate_strings(
        prediction=prediction
    )
    assert result["score"] == score

@pytest.mark.parametrize(
    "prediction, reference, score", 
    [
        (VALIDJSON, JSONSCHEMA, 1.0),
        (VALIDJSON_2, JSONSCHEMA, 0.0)
    ]
)
def test_json_schema(prediction, reference, score) -> None:
    json_schema_evaluator = load_evaluator(
        evaluator=EvaluatorType.JSON_SCHEMA_VALIDATION
    )
    result = json_schema_evaluator.evaluate_strings(
        prediction=prediction, reference=reference
    )
    assert result["score"] == score