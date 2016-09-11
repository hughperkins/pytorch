from __future__ import print_function, division
import PyTorch

from PyTorchAug import nn


def test_nnx():
    # net = nn.Minus()
    inputTensor = PyTorch.DoubleTensor(2, 3).uniform()
    print('inputTensor', inputTensor)

    PyTorch.require('nnx')
    net = nn.Minus()
    print(net.forward(inputTensor))
