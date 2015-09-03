from __future__ import print_function
import PyTorch

from PyTorchAug import *


linear = Linear(3, 5)
linear.float()
print('linear', linear)
print('linear.weight', linear.weight)
print('linear.output', linear.output)
print('linear.gradInput', linear.gradInput)

input = PyTorch.FloatTensor(4, 3).uniform()
print('input', input)
output = linear.updateOutput(input)
print('output', output)

gradInput = linear.updateGradInput(input, output)
print('gradInput', gradInput)

criterion = ClassNLLCriterion()
print('criterion', criterion)

mlp = Sequential()
print('mlp', mlp)
mlp.add(linear)

print('dir(linear)', dir(linear))

