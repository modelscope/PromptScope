from loguru import logger

from prompt_scope.core.augmentor.demonstration_augmentation.generation_by_beam_search import BeamSearchGenerationByDiversity
from prompt_scope.core.utils.sys_prompt_utils import load_json_file

if __name__ == '__main__':
    num_expand = 5
    demonstration_requirements = ""
    demonstration_dir = "examples/with_icl_examples/results"
    model_config = {
        "model_name": "qwen-plus",
        "max_tokens": 2000,
        "temperature": 1
    }
    seed_demonstration_json_pth = "examples/with_icl_examples/seed_data/seed_demonstration.json"
    seed_demonstration = load_json_file(seed_demonstration_json_pth)
    diversity_generator = BeamSearchGenerationByDiversity(
        demonstration_save_dir=demonstration_dir,
        num_expand=num_expand,
        demonstration_requirements=demonstration_requirements,
        auto_save=True,
        expand_model_config=model_config
    )
    demonstration_aug_list = diversity_generator.run(seed_demonstrations=seed_demonstration,
                                           n=10,
                                           max_steps=1,
                                           beam_width=1)
    logger.info(f"Diversity generation finished, the result is {demonstration_aug_list}")
