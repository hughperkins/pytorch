import sys
import os
import PyTorchAug
import PyTorch
import numpy as np
import theano
from theano import tensor as T

PyTorch.require('luabit')

batchSize = 2
numFrames = 4
inSize = 3
outSize = 3
kernelSize = 3

class Luabit(PyTorchAug.LuaClass):
    def __init__(self, _fromLua=False):
        self.luaclass = 'Luabit'
        if not _fromLua:
            name = self.__class__.__name__
            super(self.__class__, self).__init__([name])
        else:
            self.__dict__['__objectId'] = getNextObjectId()

luabit = Luabit()

inTensor = np.random.randn(batchSize, numFrames, inSize).astype('float32')

luaout = luabit.getOut(PyTorch.asFloatTensor(inTensor), outSize, kernelSize)

print('luaout.size()', luaout.size())
for b in range(batchSize):
  print('luaout[' + str(b) + ']', luaout[b])

