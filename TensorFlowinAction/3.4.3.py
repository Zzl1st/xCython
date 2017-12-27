#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:00:05 2017

@author: xu
"""
import tensorflow as tf
import sys
reload(sys)
sys.setdefaultencoding('utf8')

w1 = tf.Variable(tf.random_normal([2,3], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([3,1], stddev = 1, seed = 1))

x = tf.constant([[0.7,0.9]])

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#the Variable should be initialized before used
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(y))
