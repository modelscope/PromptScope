from meta_icl.core.utils import load_json_file, sav_json

data_pth = "data/场景数据/测试数据/批跑：2024-05-16 18:19:52_test_set_results.json"
data = load_json_file(data_pth)

total = len(data)
correct = 0
error_cases = []
for item in data:
    if item["intention_class"] == item["results"]["intention_class"]:
        correct += 1
    else:
        error_cases.append(item)
print("correct: \ntotal_sample: {}\ntotal_cor: {}\nrate: {}".format(total, correct, correct/total))
sav_json(error_cases, "data/场景数据/测试数据/error.json")



