#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2020/4/23 10:52
Update  on 2020/4/23 10:52
Author: 不告诉你
Software: PyCharm
GitHub: https://github.com/Saber891
"""
import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

dimensionality = 1000
max_length = 10
results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.
