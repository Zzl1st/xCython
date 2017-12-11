#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 17:59:52 2017

@author: xu
"""
import tensorflow as tf
#在名字为foo的命名控件内创建名字为v的变量
with tf.variable_scope("foo"):
    v = tf.get_variable(
            "v",[1],initializer = tf.constatnt_initializer(1.0))

#在生成上下文管理器时，将参数reuser设置为True，这样tf.get_variable函数将直接获取已经声明的变量
with tf.variable_scope("foo",reuse = True):
    v1 = tf.get_variable("v",[1])
    print v == v1