import pandas as pd

# 读取tsv文件
df = pd.read_csv('gsm_train.tsv', sep='\t')
print(df)

df = df.drop(range(10, len(df)))
print(df)

# 保存修改后的数据到新的tsv文件
df.to_csv('gsm_train.tsv', sep='\t', index=False)
