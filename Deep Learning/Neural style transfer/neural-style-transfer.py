#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2020/5/6 10:50
Update  on 2020/5/6 10:50
Author: 不告诉你
Software: PyCharm
GitHub: https://github.com/Saber891
"""
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.applications import vgg19
from keras import backend as K
from keras import layers
from scipy.optimize import fmin_l_bfgs_b
import imageio
import time

target_image_path = '1.jpg'  # 想要变换的图像的路径
style_reference_image_path = '2.jpg'  # 风格图像的路径

width, height = load_img(target_image_path).size  # 生成图像的尺寸
img_height = 400
img_width = 400


# 辅助函数
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x):
    x[:, :, 0] = x + 103.939  # vgg19.preprocess_input 的作用是减去 ImageNet 的平均像素值，使其中心为 0。这里相当于 vgg19.preprocess_input 的逆操作
    x[:, :, 1] = x + 116.779
    x[:, :, 2] = x + 123.68
    x = x[:, :, ::-1]  # 将图像由 BGR 格式转换为 RGB 格式。这也是vgg19.preprocess_input 逆操作的一部分
    x = np.clip(x, 0, 255).astype('uint8')
    return x


target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))
combination_image = K.placeholder((1, img_height, img_width, 3))  # 这个占位符用于保存生成图像

input_tensor = K.concatenate([target_image,  # 将三张图像合并为一个批量
                              style_reference_image,
                              combination_image], axis=0)

model = vgg19.VGG19(input_tensor=input_tensor,  # 利用三张图像组成的批量作为输入来构建 VGG19 网络。加载模型将使用预训练的 ImageNet 权重
                    weights='imagenet',
                    include_top=False)
print('Model loaded.')


# 内容损失
def content_loss(base, combination):
    return K.sum(K.square(combination - base))


# 风格损失
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


# 总变差损失
def total_variation_loss(x):
    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, 1:, :img_width - 1, :])
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# 定义需要最小化的最终损失
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
content_layer = 'block5_conv2'  # 用于内容损失的层
style_layers = ['block1_conv1',  # 用于风格损失的层
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1', ]
total_variation_weight = 1e-4  # 损失分量的加权平均所使用的权重
style_weight = 1.
content_weight = 0.025

loss = K.variable(0.)  # 在定义损失时将所有分量添加到这个标量变量中
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss = loss + content_weight * content_loss(target_image_features, combination_features)
for layer_name in style_layers:  # 添加每个目标层的风格损失分量
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss = loss + (style_weight / len(style_layers)) * sl

loss = loss + total_variation_weight * total_variation_loss(combination_image)  # 添加总变差损失
# 设置梯度下降过程
grads = K.gradients(loss, combination_image)[0]  # 获取损失相对于生成图像的梯度
fetch_loss_and_grads = K.function([combination_image], [loss, grads])  # 用于获取当前损失值和当前梯度值的函数


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grads_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grads_value = grads_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grads_values = np.copy(self.grads_values)
        self.loss_value = None
        self.grads_value = None
        return grads_values


evaluator = Evaluator()

result_prefix = 'my_result'
iterations = 20

x = preprocess_image(target_image_path)
x = x.flatten()
for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss,
                                     x,
                                     fprime=evaluator.grads,
                                     maxfun=20)
    print('Current loss value:', min_val)
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_%d.png' % i
    imageio.imsave(fname, img)
    print('Image saved as', fname)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
