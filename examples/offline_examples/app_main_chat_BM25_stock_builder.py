from meta_icl.core.offline.demonstration_stock_preparation.stock_builder_4_bm25 import prepare_BM25_stock


if __name__ == '__main__':
    stock_builder_config_pth = "conf/app_followup_configs/stock_builder_configs/bm25_demonstration_stock_config.yaml"
    prepare_BM25_stock(stock_builder_config_pth)



