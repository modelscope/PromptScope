from tqdm import tqdm


# from meta_icl.algorithm.PromptAgent.tasks import *
# from meta_icl.algorithm.PromptAgent.utils import *

def eval_instruction_with_loader(task, eval_prompt, base_model, dataloader, temperature=0, record_outputs=True):
    '''
        evaluate cur_prompt on task testing dataset
    '''

    build_forward_prompts_func = task.build_forward_prompts_completion
    # batch_forward_func = base_model.batch_forward_func
    call_func = base_model.call

    all_questions = []
    all_labels = []
    all_preds = []
    all_prompts = []
    all_responses = []
    eval_output = {}

    pbar = tqdm(dataloader, leave=False)
    for batch in pbar:
        batch_prompts = build_forward_prompts_func(batch['question'], eval_prompt)
        try:
            responses = [call_func(prompt=prompt).message.content for prompt in batch_prompts]
        except:
            responses = [call_func(prompt=prompt).output.text for prompt in batch_prompts]
        preds = task.batch_clean_responses(responses)
        labels = task.clean_labels(batch['answer'])
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_questions.extend(batch['question'])
        if record_outputs:
            all_prompts.extend(batch_prompts)
            all_responses.extend(responses)
        metric = task.cal_metric(all_preds, all_labels, all_questions)
        if not isinstance(metric, tuple):
            pbar.set_postfix_str(f"Test Metric: {metric:.4f}")
        else:
            pbar.set_postfix_str(f"Test Metrics: {metric}")

    if record_outputs:
        eval_output['model_inputs'] = all_prompts
        eval_output['model_responses'] = all_responses
        eval_output['preds'] = all_preds
        eval_output['labels'] = all_labels
    eval_output['correct'] = task.cal_correct(all_preds, all_labels)
    metric = task.cal_metric(all_preds, all_labels, all_questions)
    return metric, eval_output
