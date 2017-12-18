#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:00:05 2017

@author: xu
"""

#先将图片缩小一些，这样可视化能让标注框更加清楚
img_data = tf.image.resize_images(img_data,180,267,method = 1)

#tf.image.draw_bounding_boxes函数要求图像矩阵中的数字为实数，所以需要将
#图像矩阵转化为实数类型，该函数的输入是一个batch的数据，也就是多张图像
#组成的四维矩阵，所以需要将解码之后的图像矩阵加一维

batched = tf.expand_dims(tf.image.convert_image_dtype(img_data,tf.float32),0)

#给出每一张图像的所有标注框，一个标注框有四个数字，分别代表Ymin,Xmin,Ymax,Xmax
#注意！这里的数值是一个相对的倍数

boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])

#显示加入了标注框的图像
result = tf.image.draw_bounding_boxes(batched,boxes)