#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2020/4/26 15:32
Update  on 2020/4/26 15:32
Author: 不告诉你
Software: PyCharm
GitHub: https://github.com/Saber891
"""
# 使用逆序序列训练并评估一个 LSTM
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential

max_features = 10000  # 作为特征的单词个数
maxlen = 500  # 在这么多单词之后截断文本

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = [x[::-1] for x in x_train]  # 将序列反转
x_test = [x[::-1] for x in x_test]

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)  # 填充序列
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# 训练并评估一个双向 LSTM
model = Sequential()
model.add(layers.Embedding(max_features, 128))
model.add(layers.LSTM(32))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)
