from meta_icl.core.utils.utils import load_yaml_file, sav_yaml, sav_json
from meta_icl.core.utils.sys_prompt_utils import load_json_file
from easydict import EasyDict as edict
from loguru import logger
import yaml


def whether_yaml_file(config_pth):
    return config_pth.split('.')[-1] == "yaml"


def load_config(config_pth, as_edict=False):
    try:
        if config_pth.split('.')[-1] == "json":
            configs = load_json_file(config_pth)
        elif config_pth.split('.')[-1] == "yaml":
            configs = yaml.load(open(config_pth, 'r'), Loader=yaml.FullLoader)
        else:
            logger.error("currently support .json and .yaml, other types need to add load data function!")
            raise ValueError(f'{config_pth} is not a valid config file')
    except Exception as e:
        logger.error(e)
        logger.error(f'failed to load config file. {config_pth} is not a valid config file')
        configs = {}


    if as_edict:
        configs = edict(configs)

    return configs


def sav_config(config_pth, configs, is_yaml, json_copy=True):
    """

    """
    assert (config_pth.split('.')[-1] == "yaml") == is_yaml
    if is_yaml:
        sav_yaml(configs, config_pth)
    else:
        sav_json(configs, config_pth)

    if json_copy and is_yaml:
        sav_json(configs, config_pth.replace(".yaml", ".json"))


def update_icl_configs_BM25(config_pth, examples_list_pth, search_key, BM_25_index_dir):
    """

    """
    retriever_config_name = "BM25_retriever_configs"
    configs = load_config(config_pth)
    is_yaml = whether_yaml_file(config_pth)
    logger.info(f"load config from: \"{config_pth}\"")
    logger.info("previous configs: {}".format(configs))

    if "icl_configs" not in configs.keys():
        configs["icl_configs"] = {}

    if retriever_config_name not in configs["icl_configs"].keys():
        configs["icl_configs"][retriever_config_name] = {}

    configs["icl_configs"][retriever_config_name]["examples_list_pth"] = examples_list_pth
    configs["icl_configs"][retriever_config_name]["BM_25_index_dir"] = BM_25_index_dir
    configs["icl_configs"][retriever_config_name]["search_key"] = search_key
    sav_config(config_pth=config_pth, configs=configs, is_yaml=is_yaml)
    logger.info(f"save the updated config to: {config_pth}, \nupdate config as: \n{configs}\n")
    return configs


def update_icl_configs_embedding(config_pth, embedding_pth, embedding_model, examples_list_pth, search_key):
    """

    :param config_pth: predefined config pth
    :param embedding_pth: the embedding pth
    :param embedding_model: str, the embedding model name
    :param examples_list_pth: str, the pth of demonstration json file
    :param search_key: List[str],
    :return:
    """
    configs = load_config(config_pth)
    is_yaml = whether_yaml_file(config_pth)

    retriever_config_name = "embedding_retriever_configs"
    if "icl_configs" not in configs.keys():
        configs["icl_configs"] = {}

    logger.info("load the config from: {}".format(config_pth))

    try:
        logger.info("previous embedding_pth: {}\nupdated to: {}".format(
            config_pth,
            configs["icl_configs"][retriever_config_name]["embedding_pth"],
            embedding_pth))
    except:
        logger.info("Specify the embedding_pth as: {}".format(embedding_pth))

    if retriever_config_name not in configs["icl_configs"].keys():
        configs["icl_configs"][retriever_config_name] = {}

    configs["icl_configs"][retriever_config_name]["embedding_pth"] = embedding_pth

    try:
        logger.info("previous embedding_model: {}\nupdated to: {}".format(
            config_pth,
            configs["icl_configs"][retriever_config_name]["embedding_model"],
            embedding_pth))
    except:
        logger.info("Specify the embedding_model as: {}".format(embedding_model))
    configs["icl_configs"][retriever_config_name]["embedding_model"] = embedding_model

    try:
        logger.info("previous search_key: {}\nupdated to: {}".format(
            config_pth,
            configs["icl_configs"][retriever_config_name]["search_key"],
            embedding_pth))
    except:
        logger.info("Specify the embedding_model as: {}".format(embedding_model))
    configs["icl_configs"][retriever_config_name]["search_key"] = search_key

    try:
        logger.info("previous examples_pth: {}\nupdated to: {}".format(
            config_pth,
            configs["icl_configs"][retriever_config_name]["examples_pth"],
            embedding_pth))
    except:
        logger.info("Specify the examples_pth as: {}".format(examples_list_pth))
    configs["icl_configs"][retriever_config_name]["examples_pth"] = examples_list_pth

    sav_config(config_pth=config_pth, configs=configs, is_yaml=is_yaml)
    return configs
