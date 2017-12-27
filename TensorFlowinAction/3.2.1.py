#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:01:11 2017

@author: xu
"""
import tensorflow as tf
a = tf.constant([1.0,2.0],name = "a",dtype = tf.float32)
b = tf.constant([2.0,3.0],name = "b",dtype = tf.float32)
result = tf.add(a, b, name="add")
print result
