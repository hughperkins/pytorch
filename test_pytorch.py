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

print(PyTorch.FloatTensor(3,4).uniform())

print(PyTorch.FloatTensor(10,3).uniform())

