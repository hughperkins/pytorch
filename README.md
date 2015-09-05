# pytorch
POC for wrapping torch in python

# Examples

Examples of what is possible currently:
* pytorch
* pynn

Types supported currently:
* FloatTensor
* DoubleTensor
* LongTensor

(fairly easy to add others, since templated)

# Unit-tests

Run:
```
./build.sh
./run_tests.sh
```

[test](test) folder, containing test scripts
[test_output.txt](test_output.txt)

# pytorch

Run by doing:
```
./build.sh
./run.sh
```

Script:
```
from __future__ import print_function
import PyTorch
import array
import numpy

A = numpy.random.rand(6).reshape(3,2).astype(numpy.float32)
B = numpy.random.rand(8).reshape(2,4).astype(numpy.float32)

C = A.dot(B)
print('C', C)

print('calling .asTensor...')
tensorA = PyTorch.asTensor(A)
tensorB = PyTorch.asTensor(B)
print(' ... asTensor called')

print('tensorA', tensorA)

tensorA.set2d(1, 1, 56.4)
tensorA.set2d(2, 0, 76.5)
print('tensorA', tensorA)
print('A', A)

tensorA += 5
print('tensorA', tensorA)
print('A', A)

tensorA2 = tensorA + 7
print('tensorA2', tensorA2)
print('tensorA', tensorA)

tensorAB = tensorA * tensorB
print('tensorAB', tensorAB)

print('A.dot(B)', A.dot(B))

print('tensorA[2]', tensorA[2])

D = PyTorch.FloatTensor(5,3).fill(1)
print('D', D)

D[2][2] = 4
print('D', D)

D[3].fill(9)
print('D', D)

D.narrow(1,2,1).fill(0)
print('D', D)

print(PyTorch.FloatTensor(3,4).uniform())
print(PyTorch.FloatTensor(3,4).bernoulli())
print(PyTorch.FloatTensor(3,4).normal())
print(PyTorch.FloatTensor(3,4).cauchy())
print(PyTorch.FloatTensor(3,4).exponential())
print(PyTorch.FloatTensor(3,4).logNormal())
print(PyTorch.FloatTensor(3,4).geometric())
print(PyTorch.FloatTensor(3,4).geometric())
PyTorch.manualSeed(3)
print(PyTorch.FloatTensor(3,4).geometric())
PyTorch.manualSeed(3)
print(PyTorch.FloatTensor(3,4).geometric())
```

Output:

```
C [[ 0.07321505  0.21034354  0.35347611  0.30827984]
 [ 0.1136106   0.21236189  0.37396491  0.12081985]
 [ 0.1540852   0.34465638  0.59388119  0.34144905]]
calling .asTensor...
 ... asTensor called
tensorA 0.108945 0.397354
0.697115 0.0342486
0.683192 0.335684
[torch.FloatTensor of size 3x2]

tensorA 0.108945 0.397354
0.697115 56.4
76.5 0.335684
[torch.FloatTensor of size 3x2]

A [[  0.10894503   0.3973541 ]
 [  0.697115    56.40000153]
 [ 76.5          0.33568445]]
tensorA 5.10894 5.39735
5.69711 61.4
81.5 5.33568
[torch.FloatTensor of size 3x2]

A [[  5.10894489   5.39735413]
 [  5.69711494  61.40000153]
 [ 81.5          5.3356843 ]]
tensorA2 12.1089 12.3974
12.6971 68.4
88.5 12.3357
[torch.FloatTensor of size 3x2]

tensorA 5.10894 5.39735
5.69711 61.4
81.5 5.33568
[torch.FloatTensor of size 3x2]

tensorAB 1.56072 3.88211 6.61399 4.68479
9.57568 29.3573 49.0571 46.1098
13.4707 25.4291 44.7226 15.1082
[torch.FloatTensor of size 3x4]

A.dot(B) [[  1.56071877   3.88210678   6.61398697   4.68478727]
 [  9.57568169  29.35725784  49.05712891  46.10975647]
 [ 13.47066402  25.42912674  44.72264099  15.10820866]]
tensorA[2] 81.5 5.33568
[torch.FloatTensor of size 2]

D 1 1 1
1 1 1
1 1 1
1 1 1
1 1 1
[torch.FloatTensor of size 5x3]

D 1 1 1
1 1 1
1 1 4
1 1 1
1 1 1
[torch.FloatTensor of size 5x3]

D 1 1 1
1 1 1
1 1 4
9 9 9
1 1 1
[torch.FloatTensor of size 5x3]

D 1 1 0
1 1 0
1 1 0
9 9 0
1 1 0
[torch.FloatTensor of size 5x3]

0.0069189 0.0934481 0.488816 0.989025
0.125598 0.477475 0.86955 0.0472904
0.322867 0.330087 0.931925 0.639966
[torch.FloatTensor of size 3x4]

1 1 1 1
1 1 1 1
1 1 0 0
[torch.FloatTensor of size 3x4]

-1.19303 -0.351101 -1.16909 -1.03799
-0.197524 -0.812448 0.778675 -2.12981
0.42737 0.566242 -0.970235 -0.191112
[torch.FloatTensor of size 3x4]

-1.2123 -1.77004 0.0256488 -0.246493
-0.55631 2.81549 0.85101 1.73442
6.53249 -0.252495 -2.52411 -0.0164343
[torch.FloatTensor of size 3x4]

0.214561 1.79436 0.385731 0.439792
0.67608 0.434034 1.91894 1.45847
0.0678396 1.34574 1.01357 1.06901
[torch.FloatTensor of size 3x4]

0.342984 0.124501 0.019485 1.08767
0.208956 0.412723 0.702379 0.144819
0.0494364 0.158265 0.174502 1.17602
[torch.FloatTensor of size 3x4]

2 1 3 1
1 1 1 1
3 1 4 1
[torch.FloatTensor of size 3x4]

2 2 2 1
2 2 1 1
2 2 1 1
[torch.FloatTensor of size 3x4]

2 1 2 3
1 1 2 2
4 1 4 1
[torch.FloatTensor of size 3x4]

2 1 2 3
1 1 2 2
4 1 4 1
[torch.FloatTensor of size 3x4]
```

# pynn

Run by doing:
```
./build.sh
./nn_run.sh
```

Test script:
```
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

print('dir(linear)', dir(linear))

mlp = Sequential()
mlp.add(linear)

output = mlp.forward(input)
print('output', output)



import sys
sys.path.append('thirdparty/python-mnist')
from mnist import MNIST
import numpy
import array

mlp = Sequential()
linear = Linear(784, 10)
mlp.add(linear)
logSoftMax = LogSoftMax()
mlp.add(logSoftMax)
mlp.float()

criterion = ClassNLLCriterion().float()
print('got criterion')

learningRate = 0.0001

mndata = MNIST('/norep/data/mnist')
imagesList, labelsB = mndata.load_training()
images = numpy.array(imagesList).astype(numpy.float32)

labelsf = array.array('f', labelsB.tolist())
imagesTensor = PyTorch.asTensor(images)

labelsTensor = PyTorch.asTensor(labelsf)
labelsTensor += 1

desiredN = 128
maxN = int(imagesTensor.size()[0])
desiredN = min(maxN, desiredN)
imagesTensor = imagesTensor.narrow(0, 0, desiredN)
labelsTensor = labelsTensor.narrow(0, 0, desiredN)
print('imagesTensor.size()', imagesTensor.size())
print('labelsTensor.size()', labelsTensor.size())
N = int(imagesTensor.size()[0])

print('start training...')
for epoch in range(10):
    numRight = 0
    for n in range(N):
        input = imagesTensor[n]
        label = labelsTensor[n]
        labelTensor = PyTorch.FloatTensor(1)
        labelTensor[0] = label
        output = mlp.forward(input)
        prediction = PyTorch.getPrediction(output)
        if prediction == label:
            numRight += 1
        criterion.forward(output, labelTensor)
        mlp.zeroGradParameters()
        gradOutput = criterion.backward(output, labelTensor)
        mlp.backward(input, gradOutput)
        mlp.updateParameters(learningRate)
    print('epoch ' + str(epoch) + ' accuracy: ' + str(numRight * 100.0 / N) + '%')
```

Output:
```
initializing PyTorch...
generator null: False
 ... PyTorch initialized
linear nn.Linear(3 -> 5)
linear.weight 0.52919 -0.150693 -0.543619
-0.115458 -0.565938 0.50085
0.244207 0.441115 -0.255428
0.350489 -0.29339 0.495529
-0.104448 -0.201832 -0.217759
[torch.FloatTensor of size 5x3]

linear.output [torch.FloatTensor with no dimension]

linear.gradInput [torch.FloatTensor with no dimension]

input 0.0225498 0.512978 0.394288
0.360868 0.996678 0.76309
0.704499 0.350721 0.487701
0.210192 0.961437 0.38026
[torch.FloatTensor of size 4x3]

output -0.832085 0.160432 0.663604 -0.324122 -0.336708
-0.926428 0.0323411 0.865389 -0.164706 -0.549981
-0.497534 0.220308 0.734707 0.00878677 -0.395529
-0.792741 -0.122058 0.910833 -0.39688 -0.443766
[torch.FloatTensor of size 4x5]

gradInput -0.37523 0.490374 0.275895
-0.28294 0.662366 0.336923
-0.0649143 0.351637 0.28363
-0.275738 0.796327 0.0371319
[torch.FloatTensor of size 4x3]

criterion nn.ClassNLLCriterion
dir(linear) ['addBuffer', 'bias', 'gradBias', 'gradInput', 'gradWeight', 'output', 'weight']
output -0.832085 0.160432 0.663604 -0.324122 -0.336708
-0.926428 0.0323411 0.865389 -0.164706 -0.549981
-0.497534 0.220308 0.734707 0.00878677 -0.395529
-0.792741 -0.122058 0.910833 -0.39688 -0.443766
[torch.FloatTensor of size 4x5]

got criterion
imagesTensor.size() 128 784
[torch.FloatTensor of size 2]

labelsTensor.size() 128
[torch.FloatTensor of size 1]

start training...
epoch 0 accuracy: 50.0%
epoch 1 accuracy: 78.90625%
epoch 2 accuracy: 87.5%
epoch 3 accuracy: 91.40625%
epoch 4 accuracy: 98.4375%
epoch 5 accuracy: 100.0%
epoch 6 accuracy: 100.0%
epoch 7 accuracy: 100.0%
epoch 8 accuracy: 100.0%
epoch 9 accuracy: 100.0%
```

# Installation

## Pre-requisites

* Have installed torch
* Have installed 'nn'
* Have installed python (tested with 2.7, for now)
* Have installed the following python libraries:
  * numpy
  * cython
  * Jinja2

## Procedure

```
git clone https://github.com/hughperkins/pytorch.git
cd pytorch
./build.sh
```

# How to add new methods

## pytorch

* the C library methods are defined in the torch library in torch7 repo, in two files:
  * lib/TH/generic/THTensor.h
  * lib/TH/generic/THStorage.h
* simply copy the appropriate declaration into PyTorch.pyx, in the blocks that start `cdef extern from "THStorage.h":` or `cdef extern from "THTensor.h":`, as appropriate
* and add an appropriate method into FloatStorage class, or FloatTensor class
* that's it :-)

You can have a look eg at the `narrow` method as an example

Updates:
* the cython class is now called eg `_FloatTensor` instead of `FloatTensor`.  Then we create a pure Python class called `FloatTensor` around this, by inheriting from `_FloatTensor`, and providing no additional methods or properties.  The pure Python class is monkey-patchable, eg by [PyClTorch](https://github.com/hughperkins/pycltorch)
* the `cdef` properties and methods are now declared in `PyTorch.pxd`.  This means we can call these methods and properties from [PyClTorch](https://github.com/hughperkins/pycltorch)
* the `THGenerator *` object is now available, at `globalState.generator`, and used by the various random methods, eg `_FloatTensor.bernoulli()`
* modify now `PyTorch.jinja2.pyx`, instead of `PyTorch.pyx`, and similarly modify `PyTorch.jinja2.pxd` instead of `PyTorch.pxd`

## pynn

This has been simplified a bunch since before.  We no longer try to wrap C++ classes around the lua, but just directly wrap Python classes around the Lua.  The class methods and attributes are mostly generated dynamically, according to the results of querying hte lua ones.  Mostly all we have to do is create classes with appropriate names, that derive from LuaClass.  We might need to handle inheritance somehow in the future.  At the moment, this is all handled really by PyTorchAug.py.

