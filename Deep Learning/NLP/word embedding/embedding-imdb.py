#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2020/4/23 11:30
Update  on 2020/4/23 11:30
Author: 不告诉你
Software: PyCharm
GitHub: https://github.com/Saber891
"""
from keras.datasets import imdb
from keras.layers import Embedding
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

max_features = 10000  # 作为特征的单词个数
maxlen = 20  # 在这么多单词后截断文本

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)  # 将数据加载为整数列表
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)  # 将整数列表转换成形状为 (samples,maxlen) 的二维整数张量
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(10000, 8,
                    input_length=maxlen))  # 指定 Embedding 层的最大输入长度，以便后面将嵌入输入展平。 Embedding 层激活的形状为 (samples, maxlen, 8)
model.add(Flatten())  # 将三维的嵌入张量展平成形状为 (samples, maxlen * 8) 的二维张量

model.add(Dense(1, activation='sigmoid'))  # 在上面添加分类器

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)
