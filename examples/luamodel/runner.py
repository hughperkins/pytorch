"""
This will train a torch model, via the lua class in torch_model.lua

You need to have the mnist data downloaded from yann le cunn's website into ../../data/mnist
(you can do this by running the downloadmnist.sh script, from this directory)

And you need to have installed:

  pip install python-mnist
  pip install numpy
  pip install docopt

Tested on:
 - python3.4, ubuntu 15.10 64-bit

Usage:
  runner.py [options]

Options:
  --backend [cpu|cuda|cl]      backend [default: cuda]
  --numepochs NUMEPOCHS        number epochs [default: 2]
  --learningrate LEARNINGRATE  learning rate [default: 0.02]
  --batchsize BATCHSIZE        batch size [default: 32]
  --numtrain NUMTRAIN          num training examples, or -1 for all [default: -1]

"""
from __future__ import print_function, division
import sys
import os
from docopt import docopt
import PyTorch
import PyTorchHelpers
import numpy as np
from mnist import MNIST

args = docopt(__doc__)

# you can modify the following parameters:
backend = args['--backend']
batchSize = int(args['--batchsize'])
numEpochs = int(args['--numepochs'])
learningRate = float(args['--learningrate'])
numTrain = int(args['--numtrain'])

def run():
  TorchModel = PyTorchHelpers.load_lua_class('torch_model.lua', 'TorchModel')
  torchModel = TorchModel(backend, 28, 10)

  mndata = MNIST('../../data/mnist')
  imagesList, labelsList = mndata.load_training()
  labels = np.array(labelsList, dtype=np.uint8)
  images = np.array(imagesList, dtype=np.float32)
  labels += 1  # since torch/lua labels are 1-based
  N = labels.shape[0]
  print('loaded mnist training data')

  if numTrain > 0:
    N = min(N, numTrain)
  print('numExamples N', N)
  numBatches = N // batchSize
  for epoch in range(numEpochs):
    epochLoss = 0
    epochNumRight = 0
    for b in range(numBatches):
      res = torchModel.trainBatch(
        learningRate,
        images[b * batchSize:(b+1) * batchSize],
        labels[b * batchSize:(b+1) * batchSize])
#      print('res', res)
      numRight = res['numRight']
      loss = res['loss']
      epochNumRight += numRight
      epochLoss += loss
      print('epoch ' + str(epoch) + ' batch ' + str(b) + ' accuracy: ' + str(numRight * 100.0 / batchSize) + '%')
    print('epoch ' + str(epoch) + ' accuracy: ' + str(epochNumRight * 100.0 / N) + '%')

  print('finished training')
  print('loading test data...')
  imagesList, labelsList = mndata.load_testing()
  labels = np.array(labelsList, dtype=np.uint8)
  images = np.array(imagesList, dtype=np.float32)
  labels += 1  # since torch/lua labels are 1-based
  N = labels.shape[0]
  print('loaded mnist testing data')

  numBatches = N // batchSize
  epochLoss = 0
  epochNumRight = 0
  for b in range(numBatches):
    predictions = torchModel.predict(images[b * batchSize:(b+1) * batchSize]).asNumpyTensor().reshape(batchSize)
    labelsBatch = labels[b * batchSize:(b+1) * batchSize]
    numRight = (predictions == labelsBatch).sum()
    epochNumRight += numRight
  print('test results: accuracy: ' + str(epochNumRight * 100.0 / N) + '%')

if __name__ == '__main__':
  run()

