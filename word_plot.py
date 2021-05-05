# -*- coding:utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
# 构建数据
df = pd.read_csv('concat_text.csv', encoding='utf-8')
senti = df['label'].tolist()
pos = [i for i in senti if i == 1]
pos_num = len(pos)
neg = [i for i in senti if i == 0]
neg_num = len(neg)
x = ['积极情感','消极情感']
y = [pos_num, neg_num]
# 绘图
plt.bar(x=x, height=y, label='评论情感分析', color='steelblue', alpha=0.8)
# 在柱状图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
for x1, yy in zip(x, y):
    plt.text(x1, yy + 1, str(yy), ha='center', va='bottom', fontsize=20, rotation=0)
# 设置标题
plt.title("疫情信息评论情感分析")
# 为两条坐标轴设置名称
plt.xlabel("情感分类")
plt.ylabel("评论数量")
# 显示图例
plt.legend()
plt.savefig("a.jpg")
plt.show()
