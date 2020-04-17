#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2020/4/17 10:16
Update  on 2020/4/17 10:16
Author: 不告诉你
Software: PyCharm
GitHub: https://github.com/Saber891
"""
from keras import models
from keras import layers

from Demo.demo9 import mnist
from tensorflow import keras
import tensorflow as tf
import keras
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = test_images.astype('float32')/255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))