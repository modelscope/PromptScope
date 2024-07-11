#!/bin/bash

export CONFIG_FILE_PATH="/mnt1/yunze.gy/Meta-ICL/conf/ipc_configs"

python main.py \
    --prompt "帮我写一首诗。" \
    --task_description "助手是一个善于写诗的大模型。" \
    --language chinese