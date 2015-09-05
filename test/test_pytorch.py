# GENERATED FILE, do not edit by hand
# Source: test/jinja2.test_pytorch.py

from __future__ import print_function
import PyTorch
import array
import numpy





def test_pytorchDouble():
    PyTorch.manualSeed(123)
    numpy.random.seed(123)

    

    D = PyTorch.DoubleTensor(5,3).fill(1)
    print('D', D)

    D[2][2] = 4
    print('D', D)

    D[3].fill(9)
    print('D', D)

    D.narrow(1,2,1).fill(0)
    print('D', D)

    
    print(PyTorch.DoubleTensor(3,4).uniform())
    print(PyTorch.DoubleTensor(3,4).normal())
    print(PyTorch.DoubleTensor(3,4).cauchy())
    print(PyTorch.DoubleTensor(3,4).exponential())
    print(PyTorch.DoubleTensor(3,4).logNormal())
    
    print(PyTorch.DoubleTensor(3,4).bernoulli())
    print(PyTorch.DoubleTensor(3,4).geometric())
    print(PyTorch.DoubleTensor(3,4).geometric())
    PyTorch.manualSeed(3)
    print(PyTorch.DoubleTensor(3,4).geometric())
    PyTorch.manualSeed(3)
    print(PyTorch.DoubleTensor(3,4).geometric())

    print(type(PyTorch.DoubleTensor(2,3)))

    size = PyTorch.LongTensor(2)
    size[0] = 4
    size[1] = 3
    D.resize(size)
    print('D after resize:\n', D)

    print('resize1d', PyTorch.DoubleTensor().resize1d(3).fill(1))
    print('resize2d', PyTorch.DoubleTensor().resize2d(2, 3).fill(1))
    print('resize', PyTorch.DoubleTensor().resize(size).fill(1))

    def myeval(expr):
        print(expr, ':', eval(expr))

    myeval('PyTorch.DoubleTensor(3,2).nElement()')
    myeval('PyTorch.DoubleTensor().nElement()')
    myeval('PyTorch.DoubleTensor(1).nElement()')


def test_pytorchByte():
    PyTorch.manualSeed(123)
    numpy.random.seed(123)

    

    D = PyTorch.ByteTensor(5,3).fill(1)
    print('D', D)

    D[2][2] = 4
    print('D', D)

    D[3].fill(9)
    print('D', D)

    D.narrow(1,2,1).fill(0)
    print('D', D)

    
    print(PyTorch.ByteTensor(3,4).bernoulli())
    print(PyTorch.ByteTensor(3,4).geometric())
    print(PyTorch.ByteTensor(3,4).geometric())
    PyTorch.manualSeed(3)
    print(PyTorch.ByteTensor(3,4).geometric())
    PyTorch.manualSeed(3)
    print(PyTorch.ByteTensor(3,4).geometric())

    print(type(PyTorch.ByteTensor(2,3)))

    size = PyTorch.LongTensor(2)
    size[0] = 4
    size[1] = 3
    D.resize(size)
    print('D after resize:\n', D)

    print('resize1d', PyTorch.ByteTensor().resize1d(3).fill(1))
    print('resize2d', PyTorch.ByteTensor().resize2d(2, 3).fill(1))
    print('resize', PyTorch.ByteTensor().resize(size).fill(1))

    def myeval(expr):
        print(expr, ':', eval(expr))

    myeval('PyTorch.ByteTensor(3,2).nElement()')
    myeval('PyTorch.ByteTensor().nElement()')
    myeval('PyTorch.ByteTensor(1).nElement()')


def test_pytorchFloat():
    PyTorch.manualSeed(123)
    numpy.random.seed(123)

    
    A = numpy.random.rand(6).reshape(3,2).astype(numpy.float32)
    B = numpy.random.rand(8).reshape(2,4).astype(numpy.float32)

    C = A.dot(B)
    print('C', C)

    print('calling .asTensor...')
    tensorA = PyTorch.asFloatTensor(A)
    tensorB = PyTorch.asFloatTensor(B)
    print(' ... asTensor called')

    print('tensorA', tensorA)

    tensorA.set2d(1, 1, 56.4)
    tensorA.set2d(2, 0, 76.5)
    print('tensorA', tensorA)
    print('A', A)

    print('add 5 to tensorA')
    tensorA += 5
    print('tensorA', tensorA)
    print('A', A)

    print('add 7 to tensorA')
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
    print(PyTorch.FloatTensor(3,4).normal())
    print(PyTorch.FloatTensor(3,4).cauchy())
    print(PyTorch.FloatTensor(3,4).exponential())
    print(PyTorch.FloatTensor(3,4).logNormal())
    
    print(PyTorch.FloatTensor(3,4).bernoulli())
    print(PyTorch.FloatTensor(3,4).geometric())
    print(PyTorch.FloatTensor(3,4).geometric())
    PyTorch.manualSeed(3)
    print(PyTorch.FloatTensor(3,4).geometric())
    PyTorch.manualSeed(3)
    print(PyTorch.FloatTensor(3,4).geometric())

    print(type(PyTorch.FloatTensor(2,3)))

    size = PyTorch.LongTensor(2)
    size[0] = 4
    size[1] = 3
    D.resize(size)
    print('D after resize:\n', D)

    print('resize1d', PyTorch.FloatTensor().resize1d(3).fill(1))
    print('resize2d', PyTorch.FloatTensor().resize2d(2, 3).fill(1))
    print('resize', PyTorch.FloatTensor().resize(size).fill(1))

    def myeval(expr):
        print(expr, ':', eval(expr))

    myeval('PyTorch.FloatTensor(3,2).nElement()')
    myeval('PyTorch.FloatTensor().nElement()')
    myeval('PyTorch.FloatTensor(1).nElement()')


def test_pytorchLong():
    PyTorch.manualSeed(123)
    numpy.random.seed(123)

    

    D = PyTorch.LongTensor(5,3).fill(1)
    print('D', D)

    D[2][2] = 4
    print('D', D)

    D[3].fill(9)
    print('D', D)

    D.narrow(1,2,1).fill(0)
    print('D', D)

    
    print(PyTorch.LongTensor(3,4).bernoulli())
    print(PyTorch.LongTensor(3,4).geometric())
    print(PyTorch.LongTensor(3,4).geometric())
    PyTorch.manualSeed(3)
    print(PyTorch.LongTensor(3,4).geometric())
    PyTorch.manualSeed(3)
    print(PyTorch.LongTensor(3,4).geometric())

    print(type(PyTorch.LongTensor(2,3)))

    size = PyTorch.LongTensor(2)
    size[0] = 4
    size[1] = 3
    D.resize(size)
    print('D after resize:\n', D)

    print('resize1d', PyTorch.LongTensor().resize1d(3).fill(1))
    print('resize2d', PyTorch.LongTensor().resize2d(2, 3).fill(1))
    print('resize', PyTorch.LongTensor().resize(size).fill(1))

    def myeval(expr):
        print(expr, ':', eval(expr))

    myeval('PyTorch.LongTensor(3,2).nElement()')
    myeval('PyTorch.LongTensor().nElement()')
    myeval('PyTorch.LongTensor(1).nElement()')


if __name__ == '__main__':
    
    test_pytorchDouble()
    
    test_pytorchByte()
    
    test_pytorchFloat()
    
    test_pytorchLong()
    
