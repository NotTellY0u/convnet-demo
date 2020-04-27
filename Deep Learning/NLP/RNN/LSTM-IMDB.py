#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2020/4/26 13:34
Update  on 2020/4/26 13:34
Author: 不告诉你
Software: PyCharm
GitHub: https://github.com/Saber891
"""
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import imdb
from keras.preprocessing import sequence
max_features = 10000  # 作为特征的单词个数
maxlen = 500  # 在这么多单词之后截断文本
batch_size = 32

print('load data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')

input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

model =Sequential()
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer = 'rmsprop',loss = 'binary_crossentropy',metrices = ['acc'])
history = model.fit(input_train,y_train,epochs=10,batch_size=128,validation_split=0.2)
