#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:00:05 2017

@author: xu
"""

# 读取文件
#创建一个reader来读取TFRecord文件中的样例
reader = tf.TFRecordReader()
#创建一个队列来维护输入文件列表
#使用tf.train.string_input_producer()函数
filename_queue = tf.train.string_input_producer(["/home/xu/桌面/xu/TFRecord/output.tfrecords"])

#从文件中读出一个样例，也可以使用read_up_to函数一次性读取多个样例
_,serialized_example = reader.read(filename_queue)

# 解析读入的一个样例，如果需要解析多个样例，可以用parse_example函数
features = tf.parse_single_example(
    serialized_example,
    features={
        #Tensorflow提供两种不同的属性解析方法，一种方法是tf.FixedLenFeature，这种方法解析的结果为一个Tensor
        #另一种方法是tf.VarLenFeature，这种方法得到的结果为SparseTensor，用于处理稀疏数据
        'image_raw':tf.FixedLenFeature([],tf.string),
        'pixels':tf.FixedLenFeature([],tf.int64),
        'label':tf.FixedLenFeature([],tf.int64)
    })
#tf.decode_raw可以将字符串解析成图像对应的像素数组
images = tf.decode_raw(features['image_raw'],tf.uint8)
labels = tf.cast(features['label'],tf.int32)
pixels = tf.cast(features['pixels'],tf.int32)

sess = tf.Session()

# 启动多线程处理输入数据。
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

#每次运行可以读取TFRecord文件中的一个样例，当所有样例读完后，在此样例中程序会在重头读取
for i in range(10):
    image, label, pixel = sess.run([images, labels, pixels])