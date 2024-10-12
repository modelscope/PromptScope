from meta_icl.core.offline.demonstration_augmentation.generation_by_beam_search import GenerationByBeamSearch
from meta_icl.core.utils import get_current_date
from meta_icl.core.utils import load_json_file
from meta_icl.core.utils.demontration_utils import (
    demonstration_var_score,
    demonstration_expand)

if __name__ == '__main__':
    generation_config_pth = "examples/example_demonstration_aug_by_beamsearch/agent_role_followup_beam_search_examples_expansion.json"
    demonstration_dir = "logs/beam_search_results/"

    expand_config = load_json_file(generation_config_pth)
    sav_file_name = "beach_search_gen_demo_{}_{}.json".format(get_current_date(), expand_config["model_name"])
    initial_state = expand_config["initial_demonstration"]
    generator = GenerationByBeamSearch(demonstration_save_dir=demonstration_dir,
                                       demonstration_expand=demonstration_expand,
                                       demonstration_var_score=demonstration_var_score)
    generator.beam_search(max_steps=3, beam_width=3, expand_config=expand_config)
