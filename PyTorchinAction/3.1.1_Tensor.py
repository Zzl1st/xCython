#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:00:05 2017

@author: xu
"""
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import torch as py

a = torch.Tensor([[2,3],[4,8],[7,9]])
#format(a)输出张量a的数据类型
print('a is:{}'.format(a))
print('a size is{}'.format(a.size()))

#a.size()=3,2 分别输出了矩阵的行数和列数