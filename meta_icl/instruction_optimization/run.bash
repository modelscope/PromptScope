#!/bin/bash

# Define the language variable (You can set this variable to either 'english' or 'chinese' based on your need or pass it as an argument)
language=$1
q_type=$2
# Check if the language is English

if [ "$language" = "english" ]; then
    if [ "$q_type" = "classification" ]; then
        python run_pipeline.py \
            --prompt "Does this movie review contain a spoiler? answer Yes or No" \
            --task_description "Assistant is an expert classifier that will classify a movie review, and let the user know if it contains a spoiler for the reviewed movie or not." \
            --language $language \
            --num_steps 3
    elif [ "$q_type" = "generation" ]; then
        python run_generation_pipeline.py \
            --prompt "Propose a standard for scientific researchers to ranking papers in the range of 1 to 5." \
            --task_description "Assistant is a large language model that is tasked with ranking papers." \
            --language $language
            --num_steps 3
            # --load_dump '/mnt1/yunze.gy/Meta-ICL/meta_icl/ipc/dump'
    else
        echo "Unsupported question type: $q_type"
    fi

# Check if the language is Chinese
elif [ "$language" = "chinese" ]; then
    if [ "$q_type" = "classification" ]; then
        python run_pipeline.py \
            --prompt "判断计算算式的结果是否正确，回答是或者否。" \
            --task_description "判断计算算式的结果。" \
            --language $language \
            --num_steps 3
    elif [ "$q_type" = "generation" ]; then
        python run_generation_pipeline.py \
            --prompt "请为电影写一个高质量的评价。" \
            --task_description "助手是一个善于写电影评价的大模型。" \
            --language $language
    else
        echo "Unsupported question type: $q_type"
    fi
else
    echo "Unsupported language: $language"
fi