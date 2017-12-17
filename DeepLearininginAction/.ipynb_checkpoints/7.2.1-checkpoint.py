#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:00:05 2017

@author: xu
"""

import matplotlib.pyplot as plt
import tensorflow as tf   
import numpy as np

#读取图像的原始数据
image_raw_data = tf.gfile.FastGFile("/home/xu/文件/深度学习/TensorFlow/源码/tensorflow-tutorial/Deep_Learning_with_TensorFlow/datasets/cat.jpg",'r').read()

with tf.Session() as sess:
    #将图片使用jpeg的格式解码从而得到图像对应的三维矩阵，TF还提供了tf.image.decode_png函数对png格式的图像进行解码
    #解码之后的结果为一个张量，在使用它的取值之前需要明确调用运行的过程
    img_data = tf.image.decode_jpeg(image_raw_data)
    
    # 输出解码之后的三维矩阵。
    print img_data.eval()
    img_data.set_shape([1797, 2673, 3])
    print img_data.get_shape()
    
    with tf.Session() as sess:
        plt.imshow(img_data.eval())
        plt.show()