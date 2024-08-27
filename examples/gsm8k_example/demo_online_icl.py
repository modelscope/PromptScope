from meta_icl.core.online_icl.icl.ICL import EmbeddingICL
from meta_icl.core.utils.utils import load_config

from meta_icl.core.uti
def formatting_app_role_prompt(examples, query_data, configs):
    # def formatting_app_role_prompt(examples, agent_system_prompt, history_queries, last_query, num_followup=3):
    example_str = '\n\n'.join('$$智能体的性格描述$$：\n```markdown\n{agent_persona}\n````\n'
                              '$$历史对话$$:\n{history_queries}\n'
                              '$$最新一轮对话$$: \n{last_query}\n'
                              '$$你继续发送的消息$$: \n{followup}\n'.format(
        agent_persona=item["system_prompt"],
        history_queries=item["chat_history"],
        last_query=item["last_query"],
        followup=item["followup"],
    ) for item in examples)

    num_question = configs.get('num_questions')

    prompt = Instruction_Followup_Question_Rec_Role.format(
        num_question=num_question,
        example_str=example_str,
        agent_system_prompt=query_data["system_prompt"],
        history_queries=query_data["chat_history"],
        last_query=query_data["last_query"])
    return prompt

def prompt_organizing_function(cur_query: dict,
                          retrieved_examples: list,
                          task_configs: dict) -> str:



if __name__ == '__main__':
    # load online icl configs
    online_icl_config_pth = "examples/gsm8k_example/configs/gsm_online_icl_config.yaml"
    icl_configs = load_config(online_icl_config_pth)
    embedding_pth = icl_configs.get('icl_configs').get('embedding_retriever_configs').get('embedding_pth')
    task_configs = icl_configs.get('task_configs')
    example_pth = icl_configs.get('icl_configs').get('embedding_retriever_configs').get('examples_pth')
    embedding_model = icl_configs.get('icl_configs').get('embedding_retriever_configs').get('embedding_model')
    retriever_key_list = icl_configs.get('icl_configs').get('embedding_retriever_configs').get('retriever_key_list')


    icl_prompter = EmbeddingICL(embedding_pth=embedding_pth,
                                embedding_model=embedding_model,
                                examples_pth=example_pth,
                                retriever_key_list=retriever_key_list,
                                task_configs=task_configs)
    # the full query
    full_query = icl_prompter.get_meta_prompt(cur_query={'query': 'What is the capital of France?'},
                                             formatting_function=icl_configs.get('formatting_function'),
                                             num=3)

    #