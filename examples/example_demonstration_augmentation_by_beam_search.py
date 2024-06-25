from meta_icl.utils.demontration_utils import beam_search, demonstration_expand, demonstration_var_score
from meta_icl.utils.sys_prompt_utils import load_json_file, sav_json
from meta_icl.utils.utils import get_current_date
import json, os, copy

if __name__ == '__main__':
    generation_config_pth = "examples/test_examples/test_beam_search_examples_expansion.json"
    demonstration_dir = "logs/beam_search_results/"

    expand_config = load_json_file(generation_config_pth)
    sav_file_name = "beach_search_gen_demo_{}_{}.json".format(get_current_date(), expand_config["model_name"])
    initial_state = [json.dumps(expand_config["initial_demonstration"], ensure_ascii=False)]
    best_state, all_expands = beam_search(initial_state=initial_state,
                                          max_steps=3,
                                          beam_width=3,
                                          score_fn=demonstration_var_score,
                                          expand_fn=demonstration_expand,
                                          expand_fn_config=expand_config)
    print("best_state: \n{}\n\n\n".format(best_state))
    sav_json(data=all_expands, json_file_path=os.path.join(demonstration_dir, sav_file_name))
