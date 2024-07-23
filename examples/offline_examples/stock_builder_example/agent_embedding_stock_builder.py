from meta_icl.core.offline.demonstration_stock_preparation.stock_builder_4_embedding import prepare_embedding_stock
from meta_icl.core.offline.demonstration_stock_preparation.stock_builder_4_bm25 import prepare_BM25_stock


if __name__ == '__main__':
    stock_builder_config_pth = "conf/agent_followup_configs/stock_builder_configs/demonstration_embedding_stock_config.yaml"
    prepare_embedding_stock(stock_builder_config_pth)

    #
    # bm25_stock_builder_config_pth = "conf/agent_followup_configs/stock_builder_configs/bm25_demonstration_stock_config.yaml"
    # prepare_BM25_stock(bm25_stock_builder_config_pth)



