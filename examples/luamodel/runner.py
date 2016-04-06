"""
This will train a torch model, via the lua class in torch_model.lua

You need to have the mnist data downloaded from yann le cunn's website into ../../data/mnist

And you need to have installed:

  pip install python-mnist

Tested on:
 - python3.4, ubuntu 15.10 64-bit

"""
from __future__ import print_function, division
import sys
import os
import PyTorch
import PyTorchHelpers
import numpy as np
from mnist import MNIST

# you can modify the following parameters:
backend = 'cpu'  # cpu or cuda, or cl
batchSize = 32
numEpochs = 10
learningRate = 0.02

def run():
  TorchModel = PyTorchHelpers.load_lua_class('torch_model.lua', 'TorchModel')
  torchModel = TorchModel(backend, 28, 10)

  mndata = MNIST('../../data/mnist')
  imagesList, labelsList = mndata.load_training()
  print('loaded mnist training data')
  labels = np.array(labelsList, dtype=np.uint8)
  images = np.array(imagesList, dtype=np.float32)

  labels += 1  # since torch/lua labels are 1-based

  N = labels.shape[0]
  print('numExamples N', N)
  numBatches = N // batchSize
  for epoch in range(numEpochs):
    epochLoss = 0
    epochNumRight = 0
    for b in range(numBatches):
      res = torchModel.trainBatch(
        learningRate,
        PyTorch.asFloatTensor(images[b * batchSize:(b+1) * batchSize]),
        PyTorch.asByteTensor(labels[b * batchSize:(b+1) * batchSize]))
#      print('res', res)
      numRight = res['numRight']
      loss = res['loss']
      epochNumRight += numRight
      epochLoss += loss
      print('epoch ' + str(epoch) + ' batch ' + str(b) + ' accuracy: ' + str(numRight * 100.0 / batchSize) + '%')
    print('epoch ' + str(epoch) + ' accuracy: ' + str(epochNumRight * 100.0 / N) + '%')

if __name__ == '__main__':
  run()

