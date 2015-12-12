# Implemented functions

## Pytorch

### Import

```
import PyTorch
```

### Create view onto numpy arrays

```
A = numpy.random.rand(6).reshape(3,2).astype(numpy.float32)

tensorA = PyTorch.asFloatTensor(A)
```

### Basic functions

```
A = PyTorch.DoubleTensor(5,3)
A.fill(1)
print('A', A)
print('nElement', DoubleTensor(3,2).nElement())
B = A.clone()
```

### Tensor types

```
A = PyTorch.DoubleTensor(5,3).fill(1)
A = PyTorch.FloatTensor(5,3).fill(1)
A = PyTorch.LongTensor(5,3).fill(1)
A = PyTorch.ByteTensor(5,3).fill(1)
```

### Views

```
D = PyTorch.ByteTensor(5,3).fill(1)
D.narrow(1,2,1).fill(0)
```

### Per-element

```
A = PyTorch.DoubleTensor(5,3).geometric(0.9)
A += 3
A *= 3
A -= 3
A /= 3

C = A + 5
C = A - 5
C = A * 5
C = A / 2

B = PyTorch.DoubleTensor(5,3).geometric(0.9)
C = A + B
C = A - B
C = A.clone().cmul(B)
C = A / B

A += B
A -= B
A.cmul(B)
A /= B
```

### Resize

```
print('resize1d', PyTorch.DoubleTensor().resize1d(3).fill(1))
print('resize2d', PyTorch.DoubleTensor().resize2d(2, 3).fill(1))
size = PyTorch.LongTensor(2)
size[0] = 4
size[1] = 3
print('resize', PyTorch.DoubleTensor().resize(size).fill(1))
```

### Random distributions

```
PyTorch.manualSeed(123)
print(PyTorch.DoubleTensor(3,4).uniform())
print(PyTorch.DoubleTensor(3,4).normal())
print(PyTorch.DoubleTensor(3,4).cauchy())
print(PyTorch.DoubleTensor(3,4).exponential())
print(PyTorch.DoubleTensor(3,4).logNormal())

print(PyTorch.DoubleTensor(3,4).bernoulli())
print(PyTorch.DoubleTensor(3,4).geometric())
print(PyTorch.DoubleTensor(3,4).geometric())
print(PyTorch.DoubleTensor(3,4).geometric())
```

## Pynn

```
import PyTorch
from PyTorchAug import *

mlp = Sequential()
mlp.add(Linear(784, 10))
mlp.add(LogSoftMax())

crit = ClassNLLCriterion()

input = PyTorch.FloatTensor(100,784).uniform()
target = PyTorch.FloatTensor(100).fill(1)

output = input.forward(input)
loss = crit.forward(input, target)
gradOutput = crit.backward(input, target)
mlp.zeroGradParameters()
mlp.backward(input, gradOutput)
mlp.updateParameters(learningRate)
```

