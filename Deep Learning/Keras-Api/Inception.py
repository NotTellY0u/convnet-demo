#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2020/4/28 15:30
Update  on 2020/4/28 15:30
Author: 不告诉你
Software: PyCharm
GitHub: https://github.com/Saber891
"""
from keras import layers

branch_a = layers.Conv2D(128, 1,
                         activation='relu',
                         strides=2)(x)  # 每个分支都有相同的步幅值（2），这对于保持所有分支输出具有相同的尺寸是很有必要的，这样你才能将它们连接在一起
branch_b = layers.Conv2D(128, 1, activation='relu')(x)
branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)
branch_c = layers.AveragePooling2D(3, strides=2)(x)  # 在这个分支中，平均池化层用到了步幅
branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)
branch_d = layers.Conv2D(128, 1, activation='relu')(x)
branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)
output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)  # 将分支输出连接在一起，得到模块输出
