from loguru import logger
import os

from prompt_scope.core.augmentor.demonstration_augmentation.generation_by_beam_search import BeamSearchGenerationByDiversity
from prompt_scope.core.utils.sys_prompt_utils import load_json_file

if __name__ == '__main__':
    work_dir = os.path.dirname(__file__)

    num_expand = 5
    demonstration_requirements = ""
    demonstration_dir = os.path.join(work_dir, "results")
    model_config = {
        "module_name": 'dashscope_generation',
        "model_name": "qwen-plus",
        "clazz": 'models.llama_index_generation_model',
        "max_tokens": 2000,
        "seed": 1234,
        "temperature": 1
    }
    seed_demonstration_json_pth = os.path.join(work_dir, "seed_data", "seed_demonstration.json")
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
