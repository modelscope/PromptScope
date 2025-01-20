# IPC Prompt Optimization
Intent-based Prompt Calibration: Enhancing prompt optimization with synthetic boundary cases (IPC). [[Paper](https://arxiv.org/abs/2402.03099)][[Code](https://github.com/Eladlev/AutoPrompt)]

Intent-based Prompt Calibration (IPC) is designed to refine instructional prompts for language models by iteratively generating boundary cases, evaluating them, updating the prompts based on feedback, and repeating the process to enhance prompt effectiveness over multiple iterations.

- For classification tasks: Run `python examples/ipc_examples/IPC_optimization/ipc_classify.py` to optimize a prompt. The configuration file `ipc_optim_classify.yml` located in the same directory is employed. Remember to configure your labels by modifying the `label_schema`.






