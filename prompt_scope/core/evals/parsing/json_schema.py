from typing import Any, Union

from prompt_scope.core.evals.parsing.utils import parse_json_markdown

from prompt_scope.core.evals.schema import StringEvaluator


class JsonSchemaEvaluator(StringEvaluator):
    """An evaluator that validates a JSON prediction against a JSON schema reference.

    This evaluator checks if a given JSON prediction conforms to the provided JSON schema.
    If the prediction is valid, the score is True (no errors). Otherwise, the score is False (error occurred).
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the JsonSchemaEvaluator.

        Args:
            kwargs: Additional keyword arguments.

        Raises:
            ImportError: If the jsonschema package is not installed.
        """
        super().__init__()
        try:
            import jsonschema
        except ImportError:
            raise ImportError(
                "The JsonSchemaEvaluator requires the jsonschema package."
                " Please install it with `pip install jsonschema`."
            )

    @property
    def requires_input(self) -> bool:
        """This evaluator does not require input."""
        return False

    @property
    def requires_llm(self) -> bool:
        """This evaluator does not require llm."""
        return False
    
    @property
    def requires_reference(self) -> bool:
        """This evaluator requires reference."""
        return True

    @property
    def evaluation_name(self) -> str:
        """Returns the name of the evaluation."""
        return "json_schema_validation"

    def _parse_json(self, node: Any) -> Union[dict, list, None, float, bool, int, str]:
        if isinstance(node, str):
            return parse_json_markdown(node)
        elif hasattr(node, "schema") and callable(getattr(node, "schema")):
            return getattr(node, "schema")()
        return node

    def _validate(self, prediction: Any, schema: Any) -> dict:
        from jsonschema import ValidationError, validate

        try:
            validate(instance=prediction, schema=schema)
            return {
                "score": 1,
            }
        except ValidationError as e:
            return {"score": 0, "reasoning": repr(e)}

    def _evaluate_strings(
        self,
        prediction: Union[str, Any],
        input: Union[str, Any] = None,
        reference: Union[str, Any] = None,
        **kwargs: Any,
    ) -> dict:
        parsed_prediction = self._parse_json(prediction)
        schema = self._parse_json(reference)
        return self._validate(parsed_prediction, schema)
