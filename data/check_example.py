from meta_icl.utils.sys_prompt_utils import load_json_file

pth = "data/user_query_intention_examples.json"
data = load_json_file(pth)
for item in data:
    if "user_intention" not in item.keys():
        print(item)