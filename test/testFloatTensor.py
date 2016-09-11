from __future__ import print_function
import PyTorch
from test.test_helpers import myeval, myexec


def test_float_tensor():
    PyTorch.manualSeed(123)
    print('dir(G)', dir())
    print('test_float_tensor')
    a = PyTorch.FloatTensor(3, 2)
    print('got float a')
    myeval('a.dims()')
    a.uniform()
    myeval('a')
    myexec('a[1][1] = 9')
    myeval('a')
    myeval('a.size()')
