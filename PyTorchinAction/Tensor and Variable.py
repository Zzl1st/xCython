#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 17:28:07 2017

@author: xu
"""

import torch
import numpy as np

numpy_tensor = np.random.randn(10,20)
#我们可以使用下面两种方法将numpy的ndarray转化到tensor上
pytorch_tensor1 = torch.Tensor(numpy_tensor)
pytorch_tensor2 = torch.from_numpy(numpy_tensor)

#同时，我们也可以使用下面两种方法将pytorch tensor转化到numpy ndarray上
#如果pytorch tensor在cpu上
numpy_array = pytorch_tensor1.numpy()
#如果pytorch tensor在gpu上
numpy_array = pytorch_tensor1.cpu().numpy()

#我们也能够访问到Tensor的一些属性
#可以通过下面两种方式得到tensor的大小
print(pytorch_tensor1.size())