from prompt_scope.core.offline.demonstration_augmentation.generation_by_beam_search import GenerationByBeamSearch
from prompt_scope.core.utils.demontration_utils import (demonstration_var_score,
                                                    demonstration_expand)
from prompt_scope.core.utils.sys_prompt_utils import load_json_file, check_dir
from prompt_scope.core.utils.utils import get_current_date

if __name__ == '__main__':
    # generation_config_pth = ("examples/example_demonstration_aug_by_beamsearch"
    #                          "/agent_role_followup_beam_search_examples_expansion.json")
    # demonstration_dir = "logs/agent_role_beam_search_results/"
    # check_dir(demonstration_dir)
    #
    # expand_config = load_json_file(generation_config_pth)
    # sav_file_name = "beach_search_gen_demo_{}_{}.json".format(get_current_date(), expand_config["model_name"])
    # initial_state = expand_config["initial_demonstration"]
    # best_state, all_expands = beam_search(initial_state=initial_state,
    #                                       max_steps=3,
    #                                       beam_width=3,
    #                                       score_fn=demonstration_var_score,
    #                                       expand_fn=demonstration_expand,
    #                                       expand_fn_config=expand_config)
    # print("best_state: \n{}\n\n\n".format(best_state))
    # sav_json(data=all_expands, json_file_path=os.path.join(demonstration_dir, sav_file_name))

    generation_config_pth = (
        "examples/offline_examples/example_demonstration_aug_by_beamsearch/agent_role_followup_beam_search_examples_expansion.json")
    demonstration_dir = "logs/agent_role_beam_search_results/"
    check_dir(demonstration_dir)

    expand_config = load_json_file(generation_config_pth)
    sav_file_name = "beach_search_gen_demo_{}_{}.json".format(get_current_date(), expand_config["model_name"])
    initial_state = expand_config["initial_demonstration"]
    # best_state, all_expands = beam_search(initial_state=initial_state,
    #                                       max_steps=3,
    #                                       beam_width=3,
    #                                       score_fn=demonstration_var_score,
    #                                       expand_fn=demonstration_expand,
    #                                       expand_fn_config=expand_config)
    # print("best_state: \n{}\n\n\n".format(best_state))
    # sav_json(data=all_expands, json_file_path=os.path.join(demonstration_dir, sav_file_name))

    generator = GenerationByBeamSearch(demonstration_save_dir=demonstration_dir,
                                       demonstration_expand=demonstration_expand,
                                       demonstration_var_score=demonstration_var_score)
    generator.beam_search(max_steps=3, beam_width=3, expand_config=expand_config)
