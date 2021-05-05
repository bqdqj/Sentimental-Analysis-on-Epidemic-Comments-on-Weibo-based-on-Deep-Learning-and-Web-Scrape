import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
import torch
from torchtext import data
from torch.utils.data.dataset import Dataset
from torchtext.vocab import Vectors
from torchtext.data import Iterator, BucketIterator
import jieba
import re
from model import LSTM_with_Attention
from dataset import MyDataset
from text_processing import word_clean, word_cut, transform_float
import os

# 利用GPU训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
embedding_dim = 200
hidden_dim = 128
num_layers = 8
dropout = 0.2
batch_size = 8
learning_rate = 0.003
epochs = 4


# 读取训练集和测试集
train_set = pd.read_csv(r'concat_text.csv', sep=',', encoding = 'utf-8')
test_set = pd.read_csv(r'weibo_text_dir/weibo_text.csv', sep=',', encoding = 'utf-8')

test_set['text'] = test_set['text'].apply(transform_float)
test_set['text'] = test_set['text'].apply(word_clean)


# 拆分数据集为训练集和验证集
train_text, val_text, train_label, val_label = train_test_split(train_set['text'], train_set['label'], test_size=0.2, random_state=2020)

# 定义dataset的field
TEXT = data.Field(sequential=True, use_vocab=True, tokenize=word_cut, fix_length=60)
LABEL = data.Field(sequential=False, use_vocab=False)
ID = data.Field(use_vocab=False,sequential=False,dtype=torch.float16)

# 建立dataset
train = MyDataset(train_text, train_label, None, TEXT, LABEL)
val = MyDataset(val_text, val_label, None, TEXT, LABEL)
test = MyDataset(test_set['text'], None, test_set['id'], TEXT, None, ID, test=True)

vectors = Vectors(name='sgns.weibo.bigram-char', cache='./')
TEXT.build_vocab(train,vectors=vectors)

# 建立用以训练的迭代器
train_iter = data.BucketIterator(dataset=train, batch_size=8, shuffle=True, sort_within_batch=False, repeat=False)
val_iter = Iterator(val, batch_size=8, device=-1, sort=False, sort_within_batch=False, repeat=False)
test_iter = Iterator(test, batch_size=8, device=-1, sort=False, sort_within_batch=False, repeat=False)

# 建立模型
model = LSTM_with_Attention(vocab_size=len(TEXT.vocab), embedding_dim=embedding_dim,
                            hidden_dim=hidden_dim, output_dim=2,
                            n_layers=num_layers, use_bidirectional=True, use_dropout=True).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# 训练模型
model.train()
total_step = len(train_iter)
print('LSTM model is training................')
for epoch in range(0, epochs):
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()
        predicted = model(batch.text.to(device))

        loss = criterion(predicted, batch.label.to(device))
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print ('Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, epochs, i+1, total_step, loss.item()))


# 评估模型
model.eval()
print('LSTM model is testing................')
with torch.no_grad():
    correct = 0
    total = 0
for epoch, batch in enumerate(val_iter):
    predicted = model(batch.text.to(device))
    _, predicted = torch.max(predicted.data, 1)
    total += batch.label.size(0)
    correct += (predicted == batch.label.to(device)).sum().item()

print('Test Accuracy of the model: {} %'.format(100 * correct / total))

# 保存模型
torch.save(model.state_dict(), 'model.ckpt')

# 生成测试结果

output_id = []
output_label = []

with torch.no_grad():
    for i, batch in enumerate(test_iter):
        ids = batch.id
        text = batch.text.to(device)
        outputs = model(text)
        outputs = F.softmax(outputs,dim=-1)
        outputs = torch.max(outputs, 1)[1]
        output_labels = outputs.cpu().numpy()
        for j in ids:
            output_id.append(int(j.item()))
        for k in output_labels:
            output_label.append(k)


# 将测试数据的label生成csv文件
df = pd.DataFrame({'id':output_id,'label':output_label})
df = df.sort_values(by='id', ascending='True')
df.to_csv('id_label.csv')