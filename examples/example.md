# Prompt Optimization

Prompt optimization is a process of improving the quality of prompts by applying various techniques. 
Currently, we integrate three popular methods used for prompt optimization:
- Large language models as optimizers (OPRO). [[Paper](https://arxiv.org/abs/2309.03409)][[Code](https://github.com/google-deepmind/opro)]
- Intent-based Prompt Calibration: Enhancing prompt optimization with synthetic boundary cases (IPC). [[Paper](https://arxiv.org/abs/2402.03099)][[Code](https://github.com/Eladlev/AutoPrompt)]
- PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization
. [[Paper](https://arxiv.org/abs/2310.16427)][[Code](https://github.com/XinyuanWangCS/PromptAgent)]


### OPRO
OPRO (Optimization by PROmpting) is a system designed to iteratively refine and optimize prompts for language models. It orchestrates a step-by-step process to create, assess, and refine instructions or prompts targeting specific tasks within designated datasets such as MMLU, BBH, or GSM8K.

Run `python examples/opro_examples/optimize_instructions.py` to optimize a prompt. The configuration file `opro_*.yml` located in the same directory is employed.

```python
The meaning of some parameters are:
    - task_name (str): the name of task within the above dataset to search for instructions on.
    - instruction_pos (str): where to put the instruction, one of {'before_QA','Q_begin', 'Q_end', 'A_begin'}.
    - meta_prompt_type (str): the type of meta-prompt: whether to have both previous instructions and dataset exemplars (often for fine-tuned optimizers), or to have only previous instructions (often for pre-trained optimizers).
    - dataset_name (str): the name of dataset to search for instructions on.
    - module_name (str): the name of the module to use for instruction editing.
    - model_name (str): the name of the model to use for instruction editing.
    - old_instruction_score_threshold (float): only add old instructions with score no less than this threshold.
    - extract_final_answer_by_prompting_again (bool): We can often get well-formatted answer when the model has been instruction-finetuned; otherwise, we may need to prompt again with "So the final answer is" added to better extract the final answer for final parsing.
    - include_qa (bool): whether to include "Q:" and "A:" formats in the prompt.
    - evaluate_in_parallel (bool): whether to evaluate the instructions in parallel with asyncio.
    - few_shot_qa_pairs (bool): whether to have few-shot QA pairs in the meta prompt.
    - num_score_buckets (np.inf or int): the number of score buckets when we convert float accuracies to integers. Default to np.inf for not bucketizing.
    - max_num_instructions (int): the maximum number of instructions in the meta prompt.
    - meta_prompt_instructions_before_exemplars (bool): whether the instruction-score pairs are before the exemplars from the dataset.
```

### IPC

Intent-based Prompt Calibration (IPC) is designed to refine instructional prompts for language models by iteratively generating boundary cases, evaluating them, updating the prompts based on feedback, and repeating the process to enhance prompt effectiveness over multiple iterations.

- For classification tasks: Run `python examples/ipc_examples/IPC_optimization/ipc_classify.py` to optimize a prompt. The configuration file `ipc_optim_classify.yml` located in the same directory is employed. Remember to configure your labels by modifying the `label_schema`.

### PromptAgent

PromptAgent is a system designed to optimize prompts for language models. It uses a strategic planning approach to optimize prompts by iteratively selecting the best action from a set of actions based on the current state of the system. The search algorithm is chosen from MCTS and beamsearch.

Run `python examples/prompt_agent_examples/prompt_agent.py` to optimize a prompt. The configuration file `prompt_agent_*.yml` located in the same directory is employed.






