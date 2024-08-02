from meta_icl.core.offline.demonstration_stock_preparation.stock_builder_4_embedding import prepare_embedding_stock
from meta_icl.core.offline.demonstration_stock_preparation.stock_builder_4_bm25 import prepare_BM25_stock
import argparse
parser = argparse.ArgumentParser(description='Agent followups')

# 添加位置参数
parser.add_argument("--stock_builder_config_pth", type=str,
                    help="the config of stock builder", default="")

if __name__ == '__main__':
    args = parser.parse_args()
    stock_builder_config_pth = args.stock_builder_config_pth
    stock_builder_config_pth = "conf/agent_followup_configs/stock_builder_configs/demonstration_embedding_stock_config.yaml"
    prepare_embedding_stock(stock_builder_config_pth)

    #
    # bm25_stock_builder_config_pth = "conf/agent_followup_configs/stock_builder_configs/bm25_demonstration_stock_config.yaml"
    # prepare_BM25_stock(bm25_stock_builder_config_pth)



