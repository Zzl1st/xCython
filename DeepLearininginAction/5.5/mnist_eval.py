#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 12:37:53 2017

@author: xu
"""

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train

#每10秒加载一次最新的模型,并在测试数据上测试最新模型的准确率
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        #定义输入输出的格式
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        #直接使用封装好的函数来计算前向传播的结果，因为测试时不关注正则化损失的值
        #所以这里用于计算正则化损失的函数被设置为None
        y = mnist_inference.inference(x, None)
        
        #直接使用封装好的函数来计算前向传播的结果，因为测试时不关注正则化损失的值
        #所以这里用于计算正则化损失的函数被设置为None
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #通过变量重命名的方法来加载模型，这样在前向传播的过程中就不需要调用求滑动平均
        #的函数来获取平均值了。这样就可以完全共用mnist_inference.py中定义的前向传播过程
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)

#主程序
def main(argv=None):
    mnist = input_data.read_data_sets("/home/xu/文件/深度学习/TensorFlow/源码/tensorflow-tutorial/Deep_Learning_with_TensorFlow/datasets", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    main()