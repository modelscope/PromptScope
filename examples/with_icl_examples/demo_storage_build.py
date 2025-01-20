import os

from prompt_scope.core.augmentor.demonstration_storage_preparation.storage_builder_4_embedding import \
    prepare_embedding_storage

if __name__ == '__main__':
    work_dir = os.path.dirname(__file__)
    storage_builder_config_pth = os.path.join(work_dir, "configs", "demo_storage_build_config.yaml")
    examples_list_pth = os.path.join(work_dir, "results", "demonstration_2024-08-20 11:06:32_qwen-plus.json")
    icl_config_pth = os.path.join(work_dir, "configs", "gsm_online_icl_config.yaml")
    sav_dir = os.path.join(work_dir, "storage")
    prepare_embedding_storage(
        storage_builder_config_pth=storage_builder_config_pth, 
        examples_list_pth=examples_list_pth, 
        icl_config_pth=icl_config_pth, 
        sav_dir=sav_dir
        )
