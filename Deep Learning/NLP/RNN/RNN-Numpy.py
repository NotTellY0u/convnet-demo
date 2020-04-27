#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2020/4/26 10:26
Update  on 2020/4/26 10:26
Author: 不告诉你
Software: PyCharm
GitHub: https://github.com/Saber891
"""
import numpy as np

timesteps = 100  # 输入序列的时间步数
input_features = 32  # 输入特征空间的维度
output_features = 64  # 输出特征空间的维度

inputs = np.random.random((timesteps, input_features))  # 输入数据：随机噪声
state_t = np.zeros((output_features,))  # 初始状态：全零向量

W = np.random.random((output_features, input_features))  # 创建随机的权重矩阵
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_output = []
for intput_t in inputs:  # input_t 是形状为 (input_features,) 的向量
    output_t = np.tanh(np.dot(W, intput_t) + np.dot(U, state_t) + b)  # 由输入和当前状态（前一个输出）计算得到当前输出
    successive_output.append(output_t)  # 将这个输出保存到一个列表中
    state_t = output_t  # 更新网络的状态，用于下一个时间步
final_output_sequence = np.stack(successive_output, axis=0)  # 最终输出是一个形状为 (timesteps,output_features) 的二维张量