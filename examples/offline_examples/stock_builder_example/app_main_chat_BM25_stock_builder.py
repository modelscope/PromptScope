from meta_icl.core.offline.demonstration_storage_preparation.storage_builder_4_bm25 import prepare_BM25_storage


if __name__ == '__main__':
    stock_builder_config_pth = "conf/app_followup_configs/stock_builder_configs/bm25_demonstration_stock_config.yaml"
    prepare_BM25_storage(stock_builder_config_pth)



