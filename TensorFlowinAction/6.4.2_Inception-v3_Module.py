#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 00:45:01 2017

@author: xu
"""
with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                    stride = 1,padding = 'SAME'):
    
#此处省略了inception-v3模型中其其他的网络结构而直接实现最后面的inception结构，假设输入图片经过
#之前的神经网络前向传播的结果保存在变量net中
    
    with tf.variable_scope('Mixed_7c'):
        #为inception模块中每一条路径声明一个命名空间
        with tf.variable_scope('Branch_0'):
            #实现一个过滤器边长为1,深度为320的卷积层
            branch_0 = slim.conv2d(net,320,[1,1],scope = 'Conv2d_0a_1x1')
        
        #inception模块中的第二条路径，这条计算路径上的结构本身也是一个inception结构
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net,384,[1,1],scope='Conv2d_0a_1x1')
            
            #可以将多个矩阵拼接起来，第一个参数指定了拼接的维度，以下的‘3’代表了矩阵是在深度这个维度上
            #进行拼接的，而不是其他的长度和宽度维度
            branch_1 = tf.concat(3,[
                slim.conv2d(branch_1,384,[1,3],scope = 'Conv2d_ob_1x3'),
                slim.conv2d(branch_1,384,[3,1],scope = 'Conv2d_oc_3x1')])
            ]) 
            
        
        #inception模块中的第三条路径，此计算路径也是一个inception结构
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net,448,[1,1],scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2,384,[3,3],scope='Conv2d_ ob_3x3')
            branch_2 = tf.concat(3,[
                slim.conv2d(branch_2,384,
                           [1,3],scope='Conv2d_0c_1x3'),
                slim.conv2d(branch_2,384,
                           [3,1],scope='Conv2d_od_3x1')])
            
            
        #inception模块中的第四条路径
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(
            net,[3,3],scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(
            branch_3,192,[1,1],scope='Conv2d_ ob_1x1')
            
        #当前inception模块的最后输出是由上面四个计算结果拼接得到的
        net = tf.concat(3,[branch_0,branch_1,branch_2,branch_3])
            
        

    
