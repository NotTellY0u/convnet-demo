#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2020/4/17 9:19
Update  on 2020/4/17 9:19
Author: 不告诉你
Software: PyCharm
GitHub: https://github.com/Saber891
"""
import numpy as np
import tensorflow as tf

past_velocity = 0.
momentum = 0.1
while loss > 0.01:
    w,loss,gradient = get_current_parameters()
    velocity = past_velocity * momentum - leraning_rate * gradient
    w = w + momentum * velocity - learning_rate * gradient
    past_velocity = velocity
    update_parameter(w)