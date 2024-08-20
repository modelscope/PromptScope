from meta_icl.core.offline.demonstration_storage_preparation.storage_builder_4_embedding import prepare_embedding_storage
from meta_icl.core.offline.demonstration_storage_preparation.storage_builder_4_bm25 import prepare_BM25_storage


if __name__ == '__main__':
    stock_builder_config_pth = "examples/gsm8k_example/configs/demo_storage_build_config.yaml"
    prepare_embedding_storage(stock_builder_config_pth)