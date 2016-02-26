import sys
import os
import PyTorchAug
import PyTorch
import numpy as np

PyTorch.require('luabit')
class Luabit(PyTorchAug.LuaClass):
    def __init__(self, _fromLua=False):
        self.luaclass = 'Luabit'
        if not _fromLua:
            name = self.__class__.__name__
            super(self.__class__, self).__init__([name])
        else:
            self.__dict__['__objectId'] = getNextObjectId()

batchSize = 2
numFrames = 4
inSize = 3
outSize = 3
kernelSize = 3

luabit = Luabit()

inTensor = np.random.randn(batchSize, numFrames, inSize).astype('float32')
luain = PyTorch.asFloatTensor(inTensor)

luaout = luabit.getOut(luain, outSize, kernelSize)

outTensor = luaout.asNumpyTensor()
print('outTensor', outTensor)

