#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2020/4/28 11:03
Update  on 2020/4/28 11:03
Author: 不告诉你
Software: PyCharm
GitHub: https://github.com/Saber891
"""
# 用函数式 API 实现双输入问答模型
from keras.models import Model
from keras import layers
from keras import Input
import numpy as np
import keras

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

text_input = Input(shape=(None,), dtype='int32', name='text')  # 文本输入是一个长度可变的整数序列。注意，你可以选择对输入进行命名
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)  # 将输入嵌入长度为 64 的向量
encoded_text = layers.LSTM(32)(embedded_text)  # 利用 LSTM 将向量编码为单个向量
question_input = Input(shape=(None,), dtype='int32', name='question')  # 对问题进行相同的处理（使用不同的层实例）
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)
concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)  # 将编码后的问题和文本连接起来
answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)  # 在上面添加一个softmax 分类器
model = Model([text_input, question_input], answer)  # 在模型实例化时，指定两个输入和输出
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# 将数据输入到多输入模型中
num_samples = 1000
max_length = 100

text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))  # 生成虚构的 Numpy数据

question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))

answer = np.random.randint(answer_vocabulary_size, size=(num_samples))
answer = keras.utils.to_categorical(answer, answer_vocabulary_size)  # 回答是 one-hot 编码的，不是整数

model.fit([text, question], answer, epochs=10, batch_size=128)  # 使用输入组成的列表来拟合
model.fit({'text': text, 'question': question}, answer, epochs=10, batch_size=128)  # 使用输入组成的字典来拟合（只有对输入进行命名之后才能用这种方法）
