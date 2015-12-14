from __future__ import print_function
import PyTorch
import array
import numpy

A = numpy.random.rand(6).reshape(3,2).astype(numpy.float32)
tensorA = PyTorch.asFloatTensor(A)
print('type(tensorA)', type(tensorA))

