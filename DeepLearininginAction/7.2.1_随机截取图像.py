#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:00:05 2017

@author: xu
"""

boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])

#可以通过提供标注框的方式来告诉随机截取图像的算法哪些部分是“有信息量”的
begin,size,bbox_for_draw = tf.image.sample_distorted_bounding_box(
    tf.shape(img_data),bounding_boxes = boxes)

#通过标注框可视化随机截取得到的图像
batched = tf.expand_dims(
    tf.image.convert_image_dtype(img_data,tf.float32),0)
image_with_box = tf.image.draw_bounding_boxes(batched,bbox_for_draw)

distorted_image = tf.slice(img_data,begin,size)