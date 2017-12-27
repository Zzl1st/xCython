# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
from torch.autograd import Variable

x = Variable(torch.Tensor([2]),requires_grad = True)
y = x + 2
z = y ** 2 + 3
print(z)

#使用自动求导
z.backward()
print(x.grad)

a = Variable(torch.randn(10,20),requires_grad = True)
b = Variable(torch.randn(10,5),requires_grad = True)
c = Variable(torch.randn(20,5),requires_grad = True)

out = torch.mean(b - torch.matmul(a,c))
out.backward()
print(a.grad)