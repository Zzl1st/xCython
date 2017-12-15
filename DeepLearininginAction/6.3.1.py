#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 17:11:34 2017

@author: xu
"""

#我们首先要获取卷积层的参数，才能创建过滤器的权重变量和偏置项变量，卷积层的参数只与‘过滤器的尺寸及深度’以及
#‘当前层节点矩阵的深度’有关，因此声明的参数变量是一个四维矩阵，前两个维度代表了过滤器的尺寸，第三个维度代表了
#当前层的深度，第四个维度表示过滤器的深度

#共享权置
filter_weight = tf.get_variable(
        'weight',[5,5,3,16,],
        initializer = tf.truncated_normal_initializer(stddev = 0.1)]

#共享偏置
biases = tf.get_variable(
        'biases',[16],
        initializer = tf.constant_initializer(0.1))

conv = tf.nn.conv2d(input,filiter_weight,strides = [1,1,1,1],padding = 'SAME')

bias = tf.nn.bias_add(conv,biases)
actived_conc = tf.nn.relu(bias)
pool = tf.nn.max_pool(actived_conv,ksize = [1,3,3,1],stride = [1,2,2,1]),padding = 'SAME')