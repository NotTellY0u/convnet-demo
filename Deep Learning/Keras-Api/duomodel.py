#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2020/4/28 14:33
Update  on 2020/4/28 14:33
Author: 不告诉你
Software: PyCharm
GitHub: https://github.com/Saber891
"""
# 用函数式 API 实现一个三输出模型
from keras import layers
from keras import Input
from keras.models import Model

vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embeeded_post = layers.Embedding(256, vocabulary_size)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embeeded_post)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dense(128, activation='relu')(x)

age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups, activation='softmax', name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)
model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])
# 多输出模型的编译选项：多重损失
model.compile(optimizer='rmsprop',
              loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])
model.compile(optimizer='rmsprop',
              loss={'age': 'mse',
                    'income': 'categorical_crossentropy',
                    'gender': 'binary_crossentropy'})
# 多输出模型的编译选项：损失加权
model.compile(optimizer='rmsprop', loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
              loss_weight=[0.25, 1., 10.])
model.compile(optimizer='rmsprop',  # 与上述写法等效（只有输出层具有名称时才能采用这种写法）
              loss={'age': 'mse', 'income': 'categorical_crossentropy', 'gender': 'binary_crossentropy'},
              loss_weight={'age': 0.25, 'income': 1., 'gender': 10.})
# 将数据输入到多输出模型中
model.fit(posts, [age_targets, income_targets, gender_targets], epochs=10, batch_size=64)
model.fit(posts, {'age': age_targets, 'income': income_targets, 'gender': gender_targets},
          epochs=10, batch_size=64)
