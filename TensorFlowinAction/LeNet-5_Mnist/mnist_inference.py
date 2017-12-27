#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 12:37:00 2017

@author: xu
"""
import tensorflow as tf

#定义神经网络结构相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

#第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5

#第二层卷积层的尺寸和深度
CONV2_DERP = 64
CONV2_SIZE = 5

#全连接层的节点个数
FC_SIZE = 512

#通过tf.get_variable函数来获取变量
def get_weight_variable(shape,regularizer):
    weights = tf.get_variable(
            "weight",shape,
            initializer = tf.truncated_normal_initializer(stddev = 0.1))
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights
        
#定义LeNET-5神的前向传播过程
def inference(input_tensor,train,regularizer):
    #声明第一层神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
               "weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
               initializer = tf.truncated_normal_initializer(stddev = 0.1))
        conv1_biases = tf.get_variable(
                "biases",[CONV1_DEEP],initializer = tf.constant_initializer(0.0))
        #使用边长为5,深度为32的过滤器，过滤器移动的步长为1,且使用全0填充
        conv2 = tf.nn.conv2d(
                input_tensor,conv1_weights,strides = [1,1,1,1],padding = 'SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
        
    #类似的声明第二层神经网络的变量并完成前向传播过程
 