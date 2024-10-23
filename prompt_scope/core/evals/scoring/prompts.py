"""Prompts for scoring the outputs of a models for a given question.

This prompt is used to score the responses and evaluate how it follows the instructions
and answers the question. The prompt is based on the paper from
Zheng, et. al. https://arxiv.org/abs/2306.05685
"""


SCORING_PROMPT = "Please act as an impartial judge \
and evaluate the quality of the response provided by an AI \
assistant to the user question displayed below. Begin your evaluation \
by providing a short explanation. Be as objective as possible. \
After providing your explanation, you must rate the response on a scale of 1 to 10.\n\n \
The input question of the user is {input}.\n \
Now please score the following response string: {prediction}"

SCORING_PROMPT_WITH_REFERENCE = "Please act as an impartial judge \
and evaluate the quality of the response provided by an AI \
assistant to the user question displayed below. The groundtruth of the answer is {reference} \
Begin your evaluation by providing a short explanation. Be as objective as possible. \
After providing your explanation, you must rate the response on a scale of 1 to 10. \
The input question of the user is {input}.\n \
Now please score the following response string: {prediction}"

# define output format manually
# SCORING_PROMPT = "Please act as an impartial judge \
# and evaluate the quality of the response provided by an AI \
# assistant to the user question displayed below. Begin your evaluation \
# by providing a short comment. Be as objective as possible. \
# After providing your comment, you must rate the response on a scale of 1 to 10. \
# The output should be in the following json format: \
# {{\"comments\": <comments>, \
# \"score\": <score>}} \
# The input question of the user is {input}.\n \
# Now please score the following response string: {prediction}"

# SCORING_PROMPT_WITH_REFERENCE = "Please act as an impartial judge \
# and evaluate the quality of the response provided by an AI \
# assistant to the user question displayed below. The groundtruth of the answer is {reference} \
# Begin your evaluation by providing a short comment. Be as objective as possible. \
# After providing your comment, you must rate the response on a scale of 1 to 10. \
# The output should be in the following json format: \
# {{\"comments\": <comments>, \
# \"score\": <score>}} \
# The input question of the user is {input}.\n \
# Now please score the following response string: {prediction}"