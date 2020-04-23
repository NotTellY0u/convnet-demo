#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2020/4/21 16:26
Update  on 2020/4/21 16:26
Author: 不告诉你
Software: PyCharm
GitHub: https://github.com/Saber891
"""
# 使用数据增强的特征提取
import os
from keras import layers
from keras import models
from keras import optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

conv_base = VGG16(weights='imagenet',  # 初始化权重检查点
                  include_top=False,  # 是否包含密集连接分类器
                  input_shape=(150, 150, 3))  # 输入形状

base_dir = '../../dogcatdata/small'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
validation_dir = os.path.join(base_dir, 'validation')

model = models.Sequential()
model.add(conv_base)
conv_base.trainable = False
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 利用冻结的卷积基端到端的训练模型
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1. / 255)  # 注意，不能增强验证数据
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # 将所有图像的大小调整为 150×150
    batch_size=20,
    class_mode='binary')  # 因为使用了 binary_crossentropy损失，所以需要用二进制标签
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
