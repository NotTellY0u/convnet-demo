#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2020/4/26 15:32
Update  on 2020/4/26 15:32
Author: 不告诉你
Software: PyCharm
GitHub: https://github.com/Saber891
"""
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential

max_features = 10000
maxlen = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = [x[::-1] for x in x_train]
x_test = [x[::-1] for x in x_test]

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
model = Sequential()
model.add(layers.Embedding(max_features,128))
model.add(layers.LSTM(32))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer = 'rmsprop',loss = 'binary_crossentropy',metrics=['acc'])
history = model.fit(x_train,y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)
