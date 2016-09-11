from __future__ import print_function
import PyTorch
from test.test_helpers import myeval, myexec


def test_long_tensor():
    PyTorch.manualSeed(123)
    print('test_long_tensor')
    a = PyTorch.LongTensor(3, 2).geometric()
    print('a', a)
    myeval('a')
    myexec('a[1][1] = 9')
    myeval('a')
    myeval('a.size()')
    myeval('a + 2')
    myexec('a.resize2d(3,3).fill(1)')
    myeval('a')
