#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:13:47 2017

@author: xu
"""

import tensorflow as tf



ema = tf.train.ExponentialMovingAverage(0.99)
maintain_average_op = ema.apply(tf.all_variables())
#在申明滑动平均模型之后，TF会自动生成一个影子变量

for variables in tf.all_variables():
    print variables.name

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    
    sess.run(tf.assign(v,10))
    sess.run(maintain_average_op)
    
    saver.save(sess,"/home/xu/桌面/xu/5.4.1_1/ModelText.ckpt")
    print sess.run([v,ema.average(v)])