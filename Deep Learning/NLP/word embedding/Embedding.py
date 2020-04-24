#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2020/4/23 11:19
Update  on 2020/4/23 11:19
Author: 不告诉你
Software: PyCharm
GitHub: https://github.com/Saber891
"""
from keras.layers import Embedding

embedding_layer = Embedding(1000, 64)  # Embedding 层至少需要两个参数：标记的个数（这里是 1000，即最大单词索引 +1）和嵌入的维度（这里是 64）
