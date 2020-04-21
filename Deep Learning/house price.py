#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2020/4/20 11:34
Update  on 2020/4/20 11:34
Author: 不告诉你
Software: PyCharm
GitHub: https://github.com/Saber891
"""

# 回归问题

from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# 导入数据集
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
# 数据标准化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


# 模型定义
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# K折验证
k = 4
number_val_samples = len(train_data) // k
num_epochs = 100
all_score = []

for i in range(k):
    print('Processing fold #', i)
    val_date = train_data[i * number_val_samples:(i + 1) * number_val_samples]
    val_targets = train_targets[i * number_val_samples:(i + 1) * number_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * number_val_samples], train_data[(i + 1) * number_val_samples:]], axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * number_val_samples], train_targets[(i + 1) * number_val_samples:]], axis=0)

    model = build_model()

    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=1)
    val_mse, val_mae = model.evaluate(val_date, val_targets, verbose=0)
    all_score.append(val_mae)

# 修改训练循环，保存每折的验证结果
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_date = train_data[i * number_val_samples:(i + 1) * number_val_samples]  # 准备验证数据：第K个分区的数据
    val_targets = train_targets[i * number_val_samples:(i + 1) * number_val_samples]

    partial_train_data = np.concatenate(  # 准备训练数据：其他所有分区的数据
        [train_data[:i * number_val_samples],
         train_data[(i + 1) * number_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * number_val_samples],
         train_targets[(i + 1) * number_val_samples:]],
        axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_date, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=1)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)
# 计算所有轮次中的 K 折验证分数平均值
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
# 绘制验证分数
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


smooth_mae_history = smooth_curve(average_mae_history[10:])  # 删除前 10 个数据点，因为它们的取值范围与曲线上的其他点不同

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# 训练最终模型
model = build_model()
model.fit(train_data, train_targets,
          epochs=80, batch_size=16, verbose=1)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
