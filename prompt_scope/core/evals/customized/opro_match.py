## https://github.com/google-deepmind/opro/blob/main/opro/evaluation/eval_utils.py
import string
from typing import Any, List
import re
import numpy as np

from prompt_scope.core.evals.schema import StringEvaluator

# the Boolean symbols appeared in BBH tasks
BOOLEAN_SYMBOLS = [["false", "true"], ["no", "yes"], ["invalid", "valid"]]

all_lowercase_letters = string.ascii_lowercase  # "abcd...xyz"
bracketed_lowercase_letters_set = set(
    [f"({l})" for l in all_lowercase_letters]
)  # {"(a)", ...}
bracketed_uppercase_letters_set = set(
    [f"({l.upper()})" for l in all_lowercase_letters]
)  # {"(a)", ...}


def _get_index_from_symbol(answer: str) -> int:
    """Get the index from the letter symbols A, B, C, D, to extract answer texts.

    Args:
    answer (str): the string of answer like "(B)".

    Returns:
    index (int): how far the given choice is from "a", like 1 for answer "(B)".
    """
    answer = str(answer).lower()
    # extract the choice letter from within bracket
    if answer in bracketed_lowercase_letters_set:
        answer = re.findall(r"\(.*?\)", answer)[0][1]
    index = ord(answer) - ord("a")
    return index


def _get_answer_text(input_text: str, answer_symbol: str) -> str:
    """Get the text of an answer from the symbol of a multiple choice question.

    Args:
    input_text (str): the case-sensitive input or prompt that contains choice
        letters and texts, like "From which direction does the sun rise in the
        morning? (A) west (B) east (C) north (D) south". Must contain consecutive
        upper-case bracketed letters like (A) (B) (C) (D).
    answer_symbol (str): the symbol of the true answer, like "(B)" in the above
        example.

    Returns:
    answer_text (str): the text of the trueanswer, like "east" in the
    above example.
    """
    # The choice_text_list may contain the answer part "A: xxx", but it doesn't
    # matter because the index returned by _get_index_from_symbol() is unlikely
    # to be that of "A: xxx"
    re_split_string = (
            "".join([rf"\({l.upper()}\)|" for l in all_lowercase_letters]) + "A:"
    )
    choice_text_list = [
                           item.strip().lower() for item in re.split(re_split_string, input_text)
                       ][1:]
    choice_text_list = [
        re.split("\n", item)[0] for item in choice_text_list
    ]  # remove the '\n' from the text of the last choice
    # Note the input_text needs to have choice symbols in consecutive order, like
    # "(A) ... (B) ... (C) ... (D) ... (E) ..."
    answer_text = choice_text_list[_get_index_from_symbol(answer_symbol)]
    return answer_text


class OPROMatchStringEvaluator(StringEvaluator):
    """Compute an exact match between the prediction and the reference. (Implemented by OPRO)""" 

    def __init__(
        self,
        *,
        ignore_case: bool = True,
        ignore_punctuation: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation

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
        """This evaluator requires a reference."""
        return True

    @property
    def evaluation_name(self) -> str:
        """
        Get the evaluation name.

        Returns:
            str: The evaluation name.
        """
        return "opro_match"

    def _evaluate_strings(
        self,
        *,
        prediction: str,
        reference: str,
        input: str | None = None,
        treat_include_as_correct: bool,
        **kwargs: Any,
    ) -> dict:
        """
        Evaluate the exact match between the prediction and the reference.

        Args:
            reference (str): the true answer, like "(B)".
            prediction (str): the answer given in one decode, like "(A)".
            input (str): the case-sensitive input or prompt that contains choice
            letters and texts, like "From which direction does the sun rise in the
            morning? (A) west (B) east (C) north (D) south". Must contain consecutive
            upper-case bracketed letters like (A) (B) (C) (D).

        Returns:
            dict: The evaluation results containing the score.
        """

        if self.ignore_case:
            prediction = prediction.lower()
            reference = reference.lower()
        reference_included_in_prediction = reference in prediction
        if input:  # for multiple choice questions
            if reference in all_lowercase_letters:
                reference = f"({reference})"
            if prediction in all_lowercase_letters:
                prediction = f"({prediction})"
            if reference not in bracketed_lowercase_letters_set:
                return 0
            reference_text = _get_answer_text(
                input_text=input, answer_symbol=reference
            ).lower()  # 'east'
            all_symbols_raw = np.unique(re.findall(r"\([A-Z]\)", input_text))
            all_symbols = []  # to be ['(A)', '(B)', '(C)', '(D)']
            for item in sorted(list(bracketed_uppercase_letters_set)):
                if item in all_symbols_raw:
                    all_symbols.append(item)
                else:
                    break
            other_answer_texts_list = []  # ['west', 'north', 'south']
            for symbol in all_symbols:
                if _get_index_from_symbol(symbol) != _get_index_from_symbol(reference):
                    other_answer_texts_list.append(
                        _get_answer_text(input_text=input, answer_symbol=symbol)
                    )
        else:
            other_answer_texts_list = []
            reference_text = ""
        # extract the choice symbol from within bracket
        if reference in bracketed_lowercase_letters_set:
            reference = re.findall(r"\(.*?\)", reference)[0][1]  # 'b'
        if prediction in bracketed_lowercase_letters_set:
            prediction = re.findall(r"\(.*?\)", prediction)[0][1]  # 'a'

        if self.ignore_punctuation:
            prediction = prediction.translate(str.maketrans("", "", string.punctuation))
            reference = reference.translate(str.maketrans("", "", string.punctuation))
            
        result_exact_match = prediction == reference
        
        is_choice_text_exact_match = bool(input) and (
                prediction == reference_text
                )

        def _text_in_list_not_in_target(text_list, target):
            return all([item not in target for item in text_list])

        def _target_not_in_any_of_text_list(target, text_list):
            return all([target not in text for text in text_list])

        is_true_choice_text_included_and_other_choice_text_excluded = (
                bool(input)
                and reference_text in prediction
                and (  # pylint: disable=g-long-ternary
                    _text_in_list_not_in_target(
                        other_answer_texts_list, prediction.replace(reference_text, "")
                    )
                    if _target_not_in_any_of_text_list(
                        reference_text, other_answer_texts_list
                    )
                    else _text_in_list_not_in_target(other_answer_texts_list, prediction)
                )
        )
        # If the true answer is a Boolean symbol, check "Boolean match".
        is_boolean_match = False
        # import pdb;pdb.set_trace()
        if any([reference in item for item in BOOLEAN_SYMBOLS]):
            boolean_type_index = np.where(
                [reference in item for item in BOOLEAN_SYMBOLS]
            )[0][0]
            reference_as_true_or_false_str = str(
                bool(
                    np.where(
                        np.array(BOOLEAN_SYMBOLS[boolean_type_index]) == reference
                    )[0][0]
                )
            ).lower()
            if prediction in {"0", "1"}:
                prediction = str(bool(int(prediction))).lower()
            is_boolean_match = (
                    prediction == reference_as_true_or_false_str
                    or prediction.strip() == reference_as_true_or_false_str.strip()
            )

        accuracy = int(
            result_exact_match
            or is_choice_text_exact_match
            or is_true_choice_text_included_and_other_choice_text_excluded
            or is_boolean_match
        )
        if self.treat_include_as_correct:
            accuracy = int(bool(accuracy) or reference_included_in_prediction)
        return {"score": accuracy}

        
    
    
    