#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2020/4/23 10:12
Update  on 2020/4/23 10:12
Author: 不告诉你
Software: PyCharm
GitHub: https://github.com/Saber891
"""
import string
import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

characters = string.printable
token_index = dict(zip(characters,range(1, len(characters) + 1)))
max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1.
