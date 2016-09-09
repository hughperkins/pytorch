from __future__ import print_function
import PyTorch
from .test_helpers import myexec, myeval


def test_byte_tensor():
    PyTorch.manualSeed(123)
    print('test_byte_tensor')
    a = PyTorch.ByteTensor(3, 2).geometric()
    print('a', a)
    myeval('a')
    myexec('a[1][1] = 9')
    myeval('a')
    myeval('a.size()')
    myeval('a + 2')
    myexec('a.resize2d(3,3).fill(1)')
    myeval('a')
