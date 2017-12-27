#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:27:05 2017

@author: xu
"""
import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2,3],stddev = 1))
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1))

#when defining the "placeholder",the shape could not be defined,but had better
x = tf.placeholder(tf.float32, shape = (3,2), name = "input")
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#the Variable should be initialized before used
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(y,feed_dict = {x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))