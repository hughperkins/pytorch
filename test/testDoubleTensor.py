from __future__ import print_function
import PyTorch
from test.test_helpers import myeval, myexec


def test_double_tensor():
    PyTorch.manualSeed(123)
    LongTensor = PyTorch.LongTensor
    DoubleTensor = PyTorch.DoubleTensor
    LongStorage = PyTorch.LongStorage
    print('LongStorage', LongStorage)
    print('LongTensor', LongTensor)
    print('DoubleTensor', DoubleTensor)
    print('dir(G)', dir())
    print('test_double_tensor')
    a = PyTorch.DoubleTensor(3, 2)
    print('got double a')
    myeval('a.dims()')
    a.uniform()
    myeval('a')
    myexec('a[1][1] = 9')
    myeval('a')
    myeval('a.size()')
    myeval('a + 2')
    myexec('a.resize2d(3,3).fill(1)')
    myeval('a')
    myexec('size = LongStorage(2)')
    myexec('size[0] = 4')
    myexec('size[1] = 2')
    myexec('a.resize(size)')
    myeval('a')
    myeval('DoubleTensor(3,4).uniform()')
    myeval('DoubleTensor(3,4).bernoulli()')
    myeval('DoubleTensor(3,4).normal()')
    myeval('DoubleTensor(3,4).cauchy()')
    myeval('DoubleTensor(3,4).exponential()')
    myeval('DoubleTensor(3,4).logNormal()')
    myeval('DoubleTensor(3,4).geometric()')
