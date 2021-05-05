import pandas as pd
import json
import os
import re

# 正则表达式用于匹配id
pattern = re.compile('(\d+)_.*?json')

dir_path = 'CED_Dataset/original-microblog'
dir_list = os.listdir(dir_path)

rumor_num_list = []
nonrumor_num_list = []

rumor_path = 'CED_Dataset/rumor-repost'
nonrumor_path = 'CED_Dataset/non-rumor-repost'

rumor_dir_list = os.listdir(rumor_path)
nonrumor_dir_list = os.listdir(nonrumor_path)

# 创建谣言和非谣言的id列表
for r_path, n_path in zip(rumor_dir_list, nonrumor_dir_list):
    r_json_path = os.path.join(rumor_path, r_path)
    n_json_path = os.path.join(nonrumor_path, n_path)

    r_id = re.findall(pattern, r_json_path)
    n_id = re.findall(pattern, n_json_path)

    rumor_num_list.append(r_id)
    nonrumor_num_list.append(n_id)


# 建立文本信息列表和label列表
text_list = []
label_list = []
pattern = re.compile('(\d+)_.*?json')
for path in dir_list:
    json_path = os.path.join(dir_path, path)
    id = re.findall(pattern, json_path)

    if id:
        label = 1 if id in nonrumor_num_list else 0
        label_list.append(label)
        with open(json_path, 'r', encoding='UTF-8') as f:
            load_dict = json.load(f)
            text_list.append(load_dict['text'])

# 导出谣言数据csv
df = pd.DataFrame({'text':text_list, 'label':label_list})
df.to_csv('rumor_dataset.csv', encoding = 'utf-8_sig')