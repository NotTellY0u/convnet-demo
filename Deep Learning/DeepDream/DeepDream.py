#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2020/4/30 14:16
Update  on 2020/4/30 14:16
Author: 不告诉你
Software: PyCharm
GitHub: https://github.com/Saber891
"""
# 让人反胃的实现


import numpy as np
import scipy
from keras import backend as K
from keras.applications import inception_v3
from keras.preprocessing import image

K.set_learning_phase(0)  # 我们不需要训练模型，所以这个命令会禁用所有与训练有关的操作
model = inception_v3.InceptionV3(weights='imagenet',
                                 include_top=False)  # 构建不包括全连接层的 Inception V3网络。使用预训练的 ImageNet权重来加载模型
layer_contributions = {  # 这个字典将层的名称映射为一个系数，这个系数定量表示该层激活对你要最大化的损失的贡献大小。注意，
    'mixed2': 0.2,  # 层的名称硬编码在内置的 Inception V3 应用中。可以使用 model.summary() 列出所有层的名称
    'mixed3': 3.,
    'mixed4': 2.,
    'mixed5': 1.5,
}
layer_dict = dict([(layer.name, layer) for layer in model.layers])
loss = K.variable(0.)
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output

    scalling = K.prod(K.cast(K.shape(activation), 'float32'))
    loss = loss + coeff * K.sum(
        K.square(activation[:, 2:-2, 2:-2, :])) / scalling  # 将该层特征的L2范数添加到 loss 中.为了避免出现边界伪影，损失中仅包含非边界的像素

dream = model.input  # 这个张量用于保存生成的图像，即梦境图像
grads = K.gradients(loss, dream)[0]  # 计算损失相对于梦境图像的梯度
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)  # 将梯度标准化（重要技巧）
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)  # 给定一张输出图像，设置一个 Keras 函数来获取损失值和梯度值


def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grads_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at:', i, ':', loss_value)
        x += step * grads_values
    return x


step = 0.01
num_octave = 3
octave_scale = 1.4
iterations = 20

max_loss = 10.

base_image_path = '...'


def resize_img(img, size):
    img = np.copy(img)
    factors = (1, float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zooom(img, factors, order=1)


def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)


def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


img = preprocess_image(base_image_path)
original_shape = img.shape[1:3]
successive_shape = [original_shape]

for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shape.append(shape)
    successive_shape = successive_shape[::-1]
    original_img = np.copy(img)
    shrunk_original_img = resize_img(img, successive_shape[0])

    for shape in successive_shape:
        print('Processing image shape', shape)
        img = resize_img(img, shape)
        img = gradient_ascent(img,
                              iterations=iterations,
                              step=step,
                              max_loss=max_loss)
        upscaled_shrunk_original = resize_img(original_img, shape)
        save_img(img, fname='dream_at_scale_' + str(shape) + '.png')

    save_img(img, fname='final_dream_png')
