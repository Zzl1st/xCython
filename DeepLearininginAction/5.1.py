#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:34:41 2017

@author: xu
"""
import sys
reload(sys)
sys.setdefaultencoding('utf8')

#读取数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/xu/文件/TensorFlow学习笔记/源码/tensorflow-tutorial/Deep_Learning_with_TensorFlow/datasets", one_hot=True)

#数据集会自动被分成3个子集，train、validation和test。以下代码会显示数据集的大小
print "Training data size: ", mnist.train.num_examples
print "Validating data size: ", mnist.validation.num_examples
print "Testing data size: ", mnist.test.num_examples