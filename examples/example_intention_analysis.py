from meta_icl.contribs.intension_extraction import get_intention_analysis_results
from meta_icl.core.utils import load_json_file

if __name__ == '__main__':
    config_pth = "conf/base_conf.json"
    # load the config
    config = load_json_file(config_pth)

    # the current query and the chat histories
    cur_query = {
        "chat_history": [
            {
                "用户": "这个课程学习是需要下载小程序吗？",
                "客服": "可以在咱们微信小程序直接观看哦~。也可以下载 获课 app 观看学习的哈"
            },
            {
                "用户": "下载app学习是吧？。可以两个手机一起听吗？",
                "客服": "是的呢，都可以的哈。亲亲，小客服这边没有尝试过呢"
            }
        ],
        "user_query": "[笑得满地打滚]"
    }







    # the key in cur_query to do the embedding
    # embedding_key = ["user_query", "chat_history"]

    """
    get the intention analysis results, it will output a dict:
    {'user_intention': '用户想要更换已绑定的邮箱地址，当前邮箱为example@domain.com。',
    'intention_class': '更换绑定的邮箱地址'}
    
    where:
    the key 'user_intention' denotes the user's query after rewritten
    the key 'intention_class' is the user intention class.
    """

    results = get_intention_analysis_results(cur_query=cur_query,
                                             embedding_key=config["analyzer_config"]["embedding_key"],
                                             base_model=config["analyzer_config"]["base_model"],
                                             embedding_pth=config["analyzer_config"]["embedding_pth"],
                                             examples_pth=config["analyzer_config"]["examples_pth"],
                                             embedding_model=config["analyzer_config"]["embedding_model"],
                                             num=config["analyzer_config"]["topk"])
    print(results)
