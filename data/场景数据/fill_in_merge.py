from meta_icl.utils.sys_prompt_utils import load_csv, sav_json, load_json_file
import os, random
def convert_csv_2_json(data, load_key_list, sav_key_list, fill_in_keys):
    num_samples = len(data[load_key_list[0]])
    results = []

    fill_in_value = {}
    for item in fill_in_keys:
        fill_in_value[item] = None

    for idx in range(num_samples):
        tmp_data = {}
        for load_key, sav_key in zip(load_key_list, sav_key_list):
            if len(data[load_key][idx]) > 0:
                tmp_data[sav_key] = data[load_key][idx]
                if load_key in fill_in_keys:
                    fill_in_value[load_key] = data[load_key][idx]
            else:
                tmp_data[sav_key] = fill_in_value[load_key]
        results.append(tmp_data)
    return results



if __name__ == '__main__':
    # data_pth = "data/场景数据/高频问题分类.csv"
    # data = load_csv(data_pth)
    # print(data.keys())
    # load_key_list = ['\ufeff分类', "提问"]
    # sav_key_list = ["intention_class", "user_query"]
    # fill_in_keys = ['\ufeff分类']
    # results = convert_csv_2_json(data, load_key_list, sav_key_list, fill_in_keys)
    # sav_dir = "data/场景数据"
    # sav_pth = os.path.join(sav_dir, "高频问题分类.json")
    # sav_json(results, sav_pth)

    data = load_json_file("data/场景数据/测试数据/demonstration_data_single.json")
    sav_dir = "data/场景数据"
    random.shuffle(data)
    test_data = []
    demonstration_data = []
    exist_class = {}
    num_test_per_cat = 1
    for item in data:
        if item["intention_class"] not in exist_class:
            test_data.append(item)
            exist_class[item["intention_class"]] = 1
        else:
            if exist_class[item["intention_class"]] > 2:
                demonstration_data.append(item)
                # demonstration_data[-1]["chat_history"] = []
            else:
                test_data.append(item)
                exist_class[item["intention_class"]] += 1
    # sav_json(demonstration_data, os.path.join(sav_dir, '测试数据/tmp_2label_demonstration_data_single.json'))
    sav_json(test_data, os.path.join(sav_dir, '测试数据/tmp_2label_demonstration_data_single.json'))

















