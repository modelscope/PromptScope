from meta_icl.core.utils.utils import load_yaml_file
from meta_icl.core.utils.sys_prompt_utils import load_json_file
from easydict import EasyDict as edict
from loguru import logger


def load_config(config_pth, as_edict=False):
    if config_pth.split('.')[-1] == 'json':
        data = load_json_file(config_pth)
    elif config_pth.split('.')[-1] == 'yaml':
        data = load_yaml_file(config_pth)
    else:
        raise ValueError(f'{config_pth} is not a valid config file')

    if as_edict:
        data = edict(data)

    return data


def update_icl_configs(config_pth, updates_configs):
    pass




def update_icl_configs_embedding(config_pth, embedding_pth, embedding_model, examples_list_pth, search_key):
    """

    :param config_pth: prefinded config pth, with configs like: {
  "icl_configs": {
    "base_model": "Qwen_70B",
    "embedding_pth": "data/intention_analysis_examples_emb_model:<text_embedding_v1>_examples_ver_2024-05-23 09:54:08.npy",
    "examples_pth":"data/user_query_intention_examples.json",
    "topk": 3,
    "embedding_model": "text_embedding_v1"
    "search_key": "user_query"
    }
}
    :param embedding_pth: the embedding pth
    :return:
    """
    # if the config file is json file
    if config_pth.split('.')[-1] == "json":
        configs = load_json_file(config_pth)
        is_yaml = False
    elif config_pth.split('.')[-1] == "yaml":
        configs = yaml.load(open(config_pth, 'r'), Loader=yaml.FullLoader)
        is_yaml = True
    else:
        logger.error("currently support .json and .yaml, other types need to add load data function!")
        raise ValueError("currently support .json and .yaml, other types need to add load data function!")

    logger.info("load the config from: {}".format(config_pth))

    try:
        logger.info("previous embedding_pth: {}\nupdated to: {}".format(
            config_pth,
            configs["icl_configs"]["embedding_pth"],
            embedding_pth))
    except:
        logger.info("Specify the embedding_pth as: {}".format(embedding_pth))

    configs["icl_configs"]["embedding_pth"] = embedding_pth

    try:
        logger.info("previous embedding_model: {}\nupdated to: {}".format(
            config_pth,
            configs["icl_configs"]["embedding_model"],
            embedding_pth))
    except:
        logger.info("Specify the embedding_model as: {}".format(embedding_model))
    configs["icl_configs"]["embedding_model"] = embedding_model

    try:
        logger.info("previous examples_pth: {}\nupdated to: {}".format(
            config_pth,
            configs["icl_configs"]["examples_pth"],
            embedding_pth))
    except:
        logger.info("Specify the examples_pth as: {}".format(examples_list_pth))
    configs["icl_configs"]["examples_pth"] = examples_list_pth

    if is_yaml:
        sav_yaml(configs, config_pth)
    else:
        sav_json(configs, config_pth)