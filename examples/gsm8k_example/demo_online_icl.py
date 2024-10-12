from meta_icl.core.online_icl.icl.ICL import EmbeddingICL
from meta_icl.core.online_icl.icl.ICL_prompt_handler import ICLPromptHandler
from meta_icl.core.utils.config_utils import load_config

if __name__ == '__main__':
    # load online icl configs
    online_icl_config_pth = "examples/gsm8k_example/configs/gsm_online_icl_config.yaml"
    icl_configs = load_config(online_icl_config_pth)
    embedding_pth = icl_configs.get('icl_configs').get('embedding_retriever_configs').get('embedding_pth')
    task_configs = icl_configs.get('task_configs')
    example_pth = icl_configs.get('icl_configs').get('embedding_retriever_configs').get('examples_pth')
    embedding_model = icl_configs.get('icl_configs').get('embedding_retriever_configs').get('embedding_model')
    retriever_key_list = icl_configs.get('icl_configs').get('embedding_retriever_configs').get('search_key')
    prompt_pth = icl_configs.get('task_configs').get('prompt_config').get('template_path')
    language = icl_configs.get('task_configs').get('prompt_config').get('language')

    query = "3+4 = ?"

    icl_prompter = EmbeddingICL(embedding_pth=embedding_pth,
                                embedding_model=embedding_model,
                                examples_pth=example_pth,
                                retriever_key_list=retriever_key_list,
                                task_configs=task_configs)
    # the full query
    prompt_template = ICLPromptHandler(class_path=prompt_pth, language=language)
    full_query = icl_prompter.get_meta_prompt(cur_query={'input': query},
                                              formatting_function=prompt_template.organize_icl_prompt,
                                              num=3)

    results = icl_prompter.get_results(cur_query={'input': query},
                                       formatting_function=prompt_template.organize_icl_prompt,
                                       num=3)
