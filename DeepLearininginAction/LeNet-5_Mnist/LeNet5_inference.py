#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 13:59:10 2017

@author: xu
"""

import tensorflow as tf

#配置神经网络的参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

#第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5

#第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5

#全连接层的节点个数
FC_SIZE = 512

#定义CNN的Inference过程，这里新添加了一个参数train，用于区分训练过程和测试过程
#在这个程序中将用到Dropout方法，该方法可以进一步提升模型的可靠性并防止过拟合
#Dropout方法只在训练时使用
def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        
        #使用边长为5,深度为32的过滤器，过滤器移动的步长为1,且使用全0填充
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    #实现第二层池化层的inference过程，这里选用最大池化层，池化层过滤器的变长为2
    #使用全0填充并且移动的步长为0,这一层的输入是上一层的输出，也就是28×28×32的矩阵
    #其输出为14×14×32的矩阵
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

        
     #声明第三层卷积层的变量并实现前向传播过程，这一层的输入为14×14×32的矩阵
    #输出为14×14×64的矩阵
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        
        #使用边长为5,深度为64的过滤器，过滤器移动的步长为1,且使用全0填充
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

        
     #实现第四层池化层的inference过程，这一层的结构和第二层的结构是一样的，这一层的输入为
    #14×14×64的矩阵，输出为7×7×64的矩阵
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        #将第四层池化层的输出转化为第五层全连接层的输入格式，第四层的输出为7×7×64的矩阵
        #然后第五层全连接层需要的输入格式为向量，所以需要将高维矩阵拉直成一个一个向量
        #pool2.get_shape函数可以得到第四层输出矩阵的维度而不需要手工计算
        pool_shape = pool2.get_shape().as_list()
        
        #计算将矩阵拉直成向量之后的长度，等于长×宽×深
        #pool_shape[0]代表的是一个batch中数据的个数
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        
        #通过tf.reshape函数将第四层（pool2）的输出变成一个batch的向量
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

        
    #声明第五层全连接层的变量并实现前向传播过程，这一层的输入是拉直之后的一组向量
    #向量长度为3136,输出是一组长度为512的向量
    #引入了Dropout的概念，Dropout在训练时会随机将部分节点的输出改为0,可以避免过拟合问题
    #一般只在全连接层使用而不在卷积或池化层使用
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        #只有全连接层的权重需要加入正则化
        if regularizer != None: 
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

            
     #声明第六层全连接层的变量并实现前向传播过程，这一层的输入为一组长度为512的向量
    #输出为一组长度为10的向量，这一层的输出通过Softmax之后就得到了最后的分类结果
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: 
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
        
    #返回第六层的输出
    return logit
