from meta_icl.utils.demontration_utils import beam_search, demonstration_expand, demonstration_var_score
from meta_icl.utils.sys_prompt_utils import load_json_file
import json

if __name__ == '__main__':
    generation_config_pth = "examples/test_examples/test_beam_search_examples_expansion.json"
    expand_config = load_json_file(generation_config_pth)
    initial_state = [json.dumps(expand_config["initial_demonstration"], ensure_ascii=False)]
    beam_search_results = beam_search(initial_state=initial_state,
                                      max_steps=1,
                                      beam_width=3,
                                      score_fn=demonstration_var_score,
                                      expand_fn=demonstration_expand,
                                      expand_fn_config=expand_config)
    print(beam_search_results)

