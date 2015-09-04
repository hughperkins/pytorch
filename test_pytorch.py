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

print(type(PyTorch.FloatTensor(2,3)))

size = PyTorch.FloatTensor(2)
size[0] = 4
size[1] = 3
D.resize(size)
print('D after resize:\n', D)

print('resize1d', PyTorch.FloatTensor().resize1d(3).fill(-1))
print('resize2d', PyTorch.FloatTensor().resize2d(2, 3).fill(-1))
print('resize', PyTorch.FloatTensor().resize(size).fill(-1))

def myeval(expr):
    print(expr, ':', eval(expr))

myeval('PyTorch.FloatTensor(3,2).nElement()')
myeval('PyTorch.FloatTensor().nElement()')
myeval('PyTorch.FloatTensor(1).nElement()')

