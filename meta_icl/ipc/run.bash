#!/bin/bash

# Define the language variable (You can set this variable to either 'english' or 'chinese' based on your need or pass it as an argument)
language=$1

# Check if the language is English
if [ "$language" = "english" ]; then
    python run_pipeline.py \
        --prompt "Does this movie review contain a spoiler? answer Yes or No" \
        --task_description "Assistant is an expert classifier that will classify a movie review, and let the user know if it contains a spoiler for the reviewed movie or not." \
        --language $language \
        --num_steps 3

# Check if the language is Chinese
elif [ "$language" = "chinese" ]; then
    python run_pipeline.py \
        --prompt "判断计算算式的结果是否正确，回答是或者否" \
        --task_description "判断计算算式的结果" \
        --language $language \
        --num_steps 3
else
    echo "Unsupported language: $language"
fi