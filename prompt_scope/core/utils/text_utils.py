import re

def is_chinese_prompt(text):
    chinese_characters = re.findall(r'[\u4e00-\u9fff]+', text)
    chinese_characters = "".join(chinese_characters)
    if len(chinese_characters)/len(text)>0.2:
        return True
    else:
        return False
