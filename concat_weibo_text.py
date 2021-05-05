import pandas as pd
import numpy as np
import re
import jieba
from text_processing import word_clean, word_cut, transform_float
import os

dir_text = 'weibo_text_dir'

text_filenames = ['weibo_text_1.csv', 'weibo_text_2.csv',
                  'weibo_text_3.csv', 'weibo_text_4.csv',
                  'weibo_text_5.csv']

weibo_list=list()

# 读入并合并各个文微博评论文本
for i in text_filenames:
    path = os.path.join(dir_text, i)
    df = pd.read_csv(path, encoding='utf-8')
    weibo_list.append(df)

result_df = pd.concat(weibo_list, axis=0)

# 清洗微博评论文本
result_df['text'] = result_df['text'].apply(transform_float)
result_df['text'] = result_df['text'].apply(word_clean)

# 减去过短内容
result_df['text_length'] = result_df['text'].str.len()
result_df = result_df.sort_values(by='text_length', ascending='True')
result_df = result_df[result_df['text_length']>=2]

# 读入数据集并完成合并
# df_list = []
# df_path = os.path.join(dir_text, 'new_weibo_text.csv')
# df = pd.read_csv(df_path)
# df_list.append(df)
# df_list.append(result_df)
# concat_df = pd.concat(df_list, axis=0)

# # 生成csv文件
# concat_df.to_csv('concat_text.csv')
result_df.to_csv('concat_text.csv')