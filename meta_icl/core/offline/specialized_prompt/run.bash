#!/bin/bash

export CONFIG_FILE_PATH="/mnt1/yunze.gy/Meta-ICL/conf/ipc_configs"

python main.py \
    --prompt "请帮我写一份环保报告" \
    --task_description "助手是一个善于撰写工作报告的大模型。" \
    --language chinese \
    --config_file_path "/mnt1/yunze.gy/Meta-ICL/conf/ipc_configs"
