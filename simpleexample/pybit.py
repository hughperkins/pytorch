import sys
import os
import PyTorch
import PyTorchHelpers
import numpy as np

Luabit = PyTorchHelpers.load_lua_class('luabit.lua', 'Luabit')

batchSize = 2
numFrames = 4
inSize = 3
outSize = 3
kernelSize = 3

luabit = Luabit('green')
print(luabit.getName())

inTensor = np.random.randn(batchSize, numFrames, inSize).astype('float32')
luain = PyTorch.asFloatTensor(inTensor)

luaout = luabit.getOut(luain, outSize, kernelSize)

outTensor = luaout.asNumpyTensor()
print('outTensor', outTensor)

luabit.printTable({'color': 'red', 'weather': 'sunny', 'anumber': 10, 'afloat': 1.234}, 'mistletoe', {
  'row1': 'col1', 'meta': 'data'})

