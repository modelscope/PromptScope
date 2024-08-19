from meta_icl.core.utils.sys_prompt_utils import load_json_file, check_dir
from meta_icl.core.utils.utils import get_current_date
from meta_icl.core.offline.demonstration_augmentation.generation_by_beam_search import BeamSearchGenerationByDiversity

if __name__ == '__main__':
    num_expand = 5
    demonstration_generation_instruction = "请根据提供的样例，给出${num_generated_examples}个类似样例，要求和现在的样例的任务类型一致。\n\n要求：\n1. 生成的语言和提供的参考样例保持一致， 即提供的参考样例是英文的，你给出的样例也应该是英文的；如果提供的参考样例是中文的，你给出的样例也应该是中文的\n2. 给出的样例尽量与参考样例属于同一个任务类型，但和参考样例有较大区别，并且是不同domain的。\n3. 和提供的参考样例保持一致输出格式，并且每个样例必须用markdown json形式单独区分。即输出形式:\n```json\n你生成的样例1\n```\n\n```json\n你生成的样例2\n```\n\n${other_requirements}\n\n参考样例：\n${demonstration}\n\n\n请给出${num_generated_examples}个类似样例:"
    demonstration_requirements = ""
    demonstration_dir = "examples/gsm8k_example/results"
    model_config = {
        "module_name": 'dashscope_generation',
        "model_name": "qwen-plus",
        "clazz": 'models.llama_index_generation_model',
        "max_tokens": 2000,
        "seed": 1234,
        "temperature": 1
    }
    diversity_generator = BeamSearchGenerationByDiversity(
        demonstration_save_dir=demonstration_dir,
        num_expand=num_expand,
        demonstration_generation_instruction=demonstration_generation_instruction,
        demonstration_requirements=demonstration_requirements,
        seed_demonstrations=None,
        auto_save=True,
        expand_model_config=model_config
    )


    # generation_config_pth = (
    #     "examples/gsm8k_example/data/seed_demonstration.json")
    # demonstration_dir = "logs/agent_role_beam_search_results/gsm8k"
    # check_dir(demonstration_dir)
    #
    # expand_config = load_json_file(generation_config_pth)
    # sav_file_name = "beach_search_gen_demo_{}_{}.json".format(get_current_date(), expand_config["model_name"])
    # initial_state = expand_config["initial_demonstration"]
    # # best_state, all_expands = beam_search(initial_state=initial_state,
    # #                                       max_steps=3,
    # #                                       beam_width=3,
    # #                                       score_fn=demonstration_var_score,
    # #                                       expand_fn=demonstration_expand,
    # #                                       expand_fn_config=expand_config)
    # # print("best_state: \n{}\n\n\n".format(best_state))
    # # sav_json(data=all_expands, json_file_path=os.path.join(demonstration_dir, sav_file_name))
    # diversity_generator = BeamSearchGenerationByDiversity(
    #     demonstration_save_dir=demonstration_dir,
    #     num_expand=num_expand,
    #     demonstration_generation_instruction=demonstration_generation_instruction,
    #     demonstration_requirements=demonstration_requirements,
    #     seed_demonstrations=None,
    #     auto_save=True,
    #     expand_model_config=model_config
    # )
    #
    # # generator = BeamSearchGenerationByDiversity(demonstration_save_dir=demonstration_dir,
    # #                                             demonstration_expand=demonstration_expand,
    # #                                             demonstration_var_score=demonstration_var_score)
    # # generator.beam_search(max_steps=3, beam_width=3, expand_config=expand_config)
