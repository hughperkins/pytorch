from __future__ import print_function
import array
import numpy

A = numpy.random.rand(6).reshape(3,2)
B = numpy.random.rand(8).reshape(2,4)

print('A dims', len(A.shape))
for dim in range(len(A.shape)):
    print('A[' + str(dim) + ']=' + str(A.shape[dim]))

C = A.dot(B)
print('C', C)

