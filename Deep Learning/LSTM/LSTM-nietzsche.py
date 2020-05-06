#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2020/4/30 9:45
Update  on 2020/4/30 9:45
Author: 不告诉你
Software: PyCharm
GitHub: https://github.com/Saber891
"""
import keras
import numpy as np
from keras import layers
import random
import sys

path = keras.utils.get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt'
)
text = open(path).read().encode('utf-8').lower()
print('Corpus length:'), len(text)

maxlen = 60  # 提取 60 个字符组成的序列
step = 3  # 每 3 个字符采样一个新序列
sentenses = []  # 保存所提取的序列
next_chars = []  # 保存目标（即下一个字符）
for i in range(0, len(text) - maxlen, step):
    sentenses.append(text[i:i + maxlen])
    next_chars.append(text[i + maxlen])
print('Number of sequences:', len(sentenses))
chars = sorted(list(set(text)))  # 语料中唯一字符组成的列表
print('Unique characters:', len(chars))
char_indics = dict((char, chars.index(char)) for char in chars)  # 一个字典，将唯一字符映射为它在列表 chars 中的索引
print('Vectorization...')
x = np.zeros((len(sentenses), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentenses), len(chars)), dtype=np.bool)
for i, sentense in enumerate(sentenses):  # 将字符 one-hot 编码为二进制数组
    for t, char in enumerate(sentense):
        x[i, t, char_indics[char]] = 1
    y[i, char_indics[next_chars[i]]] = 1
# 用于预测下一个字符的单层 LSTM 模型
model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))
# 模型编译配置
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# 给定模型预测，采样下一个字符的函数
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# 文本生成循环
for epoch in range(1, 60):  # 将模型训练 60 轮
    print('epoch', epoch)
    model.fit(x, y, batch_size=128, epochs=1)  # 将模型在数据上拟合一次
    start_index = random.randint(0, len(text) - maxlen - 1)  # 随机选择一个文本种子
    generated_text = text[start_index:start_index + maxlen]
    print('-- Generating with seed:"' + generated_text + '"')
    for temperature in [0.2, 0.5, 1.0, 1.2]:  # 尝试一系列不同的采样温度
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)

        for i in range(400):  # 从种子文本开始，生成400个字符
            sampled = np.zeros((1, maxlen, len(chars)))  # 对目前生成的字符进行one-hot 编码
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indics[char]] = 1

            preds = model.predict(sampled, verbose=0)[0]  # 对下一个字符进行采样
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            sys.stdout.write(next_char)
