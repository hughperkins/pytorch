# pytorch
Wrappers to use torch and lua from python

# What is pytorch?

- create torch tensors, call operations on them
- instantiate `nn` network modules, train them, make predictions
- create your own lua class, call methods on that

## Create torch tensors

```
import PyTorch
a = PyTorch.FloatTensor(2,3).uniform()
a += 3
print('a', a)
print('a.sum()', a.sum())
```

## Instantiate nn network modules

```
import PyTorch
from PyTorchAug import nn

net = nn.Sequential()
net.add(nn.SpatialConvolutionMM(1, 16, 5, 5, 1, 1, 2, 2))
net.add(nn.ReLU())
net.add(nn.SpatialMaxPooling(3, 3, 3, 3))

net.add(nn.SpatialConvolutionMM(16, 32, 3, 3, 1, 1, 1, 1))
net.add(nn.ReLU())
net.add(nn.SpatialMaxPooling(2, 2, 2, 2))

net.add(nn.Reshape(32 * 4 * 4))
net.add(nn.Linear(32 * 4 * 4, 150))
net.add(nn.Tanh())
net.add(nn.Linear(150, 10))
net.add(nn.LogSoftMax())
net.float()

crit = nn.ClassNLLCriterion()
crit.float()

net.zeroGradParameters()
input = PyTorch.FloatTensor(5, 1, 28, 28).uniform()
labels = PyTorch.ByteTensor(5).geometric(0.9).icmin(10)
output = net.forward(input)
loss = crit.forward(output, labels)
gradOutput = crit.backward(output, labels)
gradInput = net.backward(input, gradOutput)
net.updateParameters(0.02)
```

# Write your own lua class, call methods on it

Example lua class:
```
require 'torch'
require 'nn'

local TorchModel = torch.class('TorchModel')

function TorchModel:__init(backend, imageSize, numClasses)
  self:buildModel(backend, imageSize, numClasses)
  self.imageSize = imageSize
  self.numClasses = numClasses
  self.backend = backend
end

function TorchModel:buildModel(backend, imageSize, numClasses)
  self.net = nn.Sequential()
  local net = self.net

  net:add(nn.SpatialConvolutionMM(1, 16, 5, 5, 1, 1, 2, 2))
  net:add(nn.ReLU())
  net:add(nn.SpatialMaxPooling(3, 3, 3, 3))
  net:add(nn.SpatialConvolutionMM(16, 32, 3, 3, 1, 1, 1, 1))
  net:add(nn.ReLU())
  net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  net:add(nn.Reshape(32 * 4 * 4))
  net:add(nn.Linear(32 * 4 * 4, 150))
  net:add(nn.Tanh())
  net:add(nn.Linear(150, numClasses))
  net:add(nn.LogSoftMax())

  self.crit = nn.ClassNLLCriterion()

  self.net:float()
  self.crit:float()
end

function TorchModel:trainBatch(learningRate, input, labels)
  self.net:zeroGradParameters()

  local output = self.net:forward(input)
  local loss = self.crit:forward(output, labels)
  local gradOutput = self.crit:backward(output, labels)
  self.net:backward(input, gradOutput)
  self.net:updateParameters(learningRate)

  local _, prediction = output:max(2)
  local numRight = labels:int():eq(prediction:int()):sum()
  return {loss=loss, numRight=numRight}  -- you can return a table, it will become a python dictionary
end

function TorchModel:predict(input)
  local output = self.net:forward(input)
  local _, prediction = output:max(2)
  return prediction:byte()
end
```

Python script that calls this.  Assume the lua class is stored in file "torch_model.lua"
```
import PyTorch
import PyTorchHelpers
import numpy as np
from mnist import MNIST

batchSize = 32
numEpochs = 2
learningRate = 0.02

TorchModel = PyTorchHelpers.load_lua_class('torch_model.lua', 'TorchModel')
torchModel = TorchModel(backend, 28, 10)

mndata = MNIST('../../data/mnist')
imagesList, labelsList = mndata.load_training()
labels = np.array(labelsList, dtype=np.uint8)
images = np.array(imagesList, dtype=np.float32)
labels += 1  # since torch/lua labels are 1-based
N = labels.shape[0]

numBatches = N // batchSize
for epoch in range(numEpochs):
  epochLoss = 0
  epochNumRight = 0
  for b in range(numBatches):
    res = torchModel.trainBatch(
      learningRate,
      images[b * batchSize:(b+1) * batchSize],
      labels[b * batchSize:(b+1) * batchSize])
    numRight = res['numRight']
    epochNumRight += numRight
  print('epoch ' + str(epoch) + ' accuracy: ' + str(epochNumRight * 100.0 / N) + '%')
```

It's easy to modify the lua script to use CUDA, or OpenCL.

# Installation

## Pre-requisites

* Have installed torch, following instructions at [https://github.com/torch/distro](https://github.com/torch/distro)
* Have installed 'nn' torch module:
```
luarocks install nn
```
* Have installed python (tested with 2.7 and 3.4)
* lua51 headers should be installed, ie something like `sudo apt-get install lua5.1 liblua5.1-dev`
Run:
```
pip install -r requirements.txt
```
* To be able to run tests, also do:
```
pip install -r test/requirements.txt
```

## Procedure

Run:
```
git clone https://github.com/hughperkins/pytorch.git
cd pytorch
source ~/torch/install/bin/torch-activate
./build.sh
```

# Unit-tests

Run:
```
source ~/torch/install/bin/torch-activate
cd pytorch
./run_tests.sh
```

# Python 2 vs Python 3?

- pytorch is developed and maintained on python 3
- you should be able to use it with python 2, but there might be the occasional oversight.  Please log an issue
for any python 2 incompatibilities you notice

# Maintainer guidelines

[Maintainer guidelines](doc/Maintainer_guidelines.md)

# Versioning

[semantic versioning](http://semver.org/)

# Related projects

Examples of training models/networks using pytorch:
* [pytorch-residual-networks](https://github.com/hughperkins/pytorch-residual-networks) port of Michael Wilber's [torch-residual-networks](https://github.com/gcr/torch-residual-networks), to handle data loading and preprocessing from Python, via pytorch
* [cifar.pytorch](https://github.com/hughperkins/cifar.pytorch) pytorch implementation of Sergey's [cifar.torch](https://github.com/szagoruyko/cifar.torch)

Addons, for using cuda tensors and opencl tensors directly from python (no need for this to train networks.  could be useful if you want to manipulate cuda tensor
directly from python)
* [pycltorch](https://github.com/hughperkins/pycltorch) python wrappers for [cltorch](https://github.com/hughperkins/cltorch) and [clnn](https://github.com/hughperkins/clnn)
* [pycudatorch](https://github.com/hughperkins/pycudatorch) python wrappers for [cutorch](https://github.com/torch/cutorch) and [cunn](https://github.com/torch/cunn)

# Recent news

12 September:
* Yannick Hold-Geoffroy added conversion of lists and tuples to Lua tables

8 September:
* added `PyTorchAug.save(filename, object)` and `PyTorchAug.load(filename)`, to save/load Torch `.t7` files

26 August:
* if not deploying to a virtual environment, will install with `--user`, into home directory

14 April:
* stack trace should be a bit more useful now :-)

17 March:
* ctrl-c works now (tested on linux)

16 March:
* uses luajit on linux now (mac os x continues to use lua)

6 March:
* all classes should be usable from `nn` now, without needing to explicitly register inside `pytorch`
  * you need to upgrade to `v3.0.0` to enable this, which is a breaking change, since the `nn` classes are now in `PyTorchAug.nn`, instead of directly
in `PyTorchAug`

5 March:
* added `PyTorchHelpers.load_lua_class(lua_filename, lua_classname)` to easily import a lua class from a lua file
* can pass parameters to lua class constructors, from python
* can pass tables to lua functions, from python (pass in as python dictionaries, become lua tables)
* can return tables from lua functions, to python (returned as python dictionaries)

2 March:
* removed requirements on Cython, Jinja2 for installation

28th Februrary:
* builds ok on Mac OS X now :-)  See https://travis-ci.org/hughperkins/pytorch/builds/112292866

26th February:
* modified `/` to be the div operation for float and double tensors, and `//` for int-type tensors, such as
byte, long, int
* since the div change is incompatible with 1.0.0 div operators, jumping radically from `1.0.0` to `2.0.0-SNAPSHOT` ...
* added dependency on `numpy`
* added `.asNumpyTensor()` to convert a torch tensor to a numpy tensor

24th February:
* added support for passing strings to methods
* added `require`
* created prototype for importing your own classes, and calling methods on those
* works with Python 3 now :-)

[Older changes](doc/oldchanges.md)

