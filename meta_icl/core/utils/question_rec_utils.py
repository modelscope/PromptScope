import re
from typing import List


def extract_feedbacks(text: str) -> List[str]:
    """
    using regular express match the content between <START> and <END>
    :param text: str
    :return: list of str
    """
    # using regular express match the content between <START> and <END>
    matches = re.findall(r'<START>(.*?)<END>', text, re.DOTALL)

    # put each content into a list
    suggestions = [match.strip() for match in matches]
    return suggestions
