#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2020/4/23 15:02
Update  on 2020/4/23 15:02
Author: 不告诉你
Software: PyCharm
GitHub: https://github.com/Saber891
"""
import numpy as np
a = np.array([[1,2],[3,5]])
y = np.expand_dims(a, axis=2)
z = np.expand_dims(a, axis=1)
print('a:',a)
print('y：',y)
print('z:',z)
print(a.shape)
print(y.shape)
print(z.shape)
