from __future__ import print_function
import PyTorch
import array
import numpy

A = numpy.random.rand(6).reshape(3,2).astype(numpy.float32)
B = numpy.random.rand(8).reshape(2,4).astype(numpy.float32)

C = A.dot(B)
print('C', C)

anArray = array.array('f', [3] * 2 * 3)
res = PyTorch.process1(2, 3, anArray)
print('res', res)

res = PyTorch.process1(2, 3, A.reshape(6))
print('res', res)

res = PyTorch.process2(A)
print('res', res)
print('A', A)

PyTorch.process3(A)
print('A', A)

print('calling .asTensor...')
tensorA = PyTorch.asTensor(A)
tensorB = PyTorch.asTensor(B)
print(' ... asTensor called')

print('torch tensor get2d:')
for r in range(3):
    thisline = ''
    for c in range(2):
        thisline += ' ' + str(tensorA.get2d(r,c))
    print(thisline)
print('')

tensorA.set2d(1, 1, 56.4)
tensorA.set2d(2, 0, 76.5)
print('A', A)

tensorA += 5
print('A', A)

tensorA2 = tensorA + 7
print('tensorA2 get2d:')
for r in range(3):
    thisline = ''
    for c in range(2):
        thisline += ' ' + str(tensorA2.get2d(r,c))
    print(thisline)
print('')

print('tensorA get2d:')
for r in range(3):
    thisline = ''
    for c in range(2):
        thisline += ' ' + str(tensorA.get2d(r,c))
    print(thisline)
print('')
#print('tensorA2', tensorA2)

tensorAB = tensorA * tensorB
print('tensorAB get2d:')
for r in range(3):
    thisline = ''
    for c in range(4):
        thisline += ' ' + str(tensorAB.get2d(r,c))
    print(thisline)
print('')
#print('tensorAB', tensorAB)

print('A.dot(B)', A.dot(B))

# A = array.array('f', [0] * (N * planes * size * size))

# PyTorch.go()


