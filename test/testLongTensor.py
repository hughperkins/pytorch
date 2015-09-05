from __future__ import print_function

import PyTorch
import inspect

def myeval(expr):
    parent_vars = inspect.stack()[1][0].f_locals
    print(expr, ':', eval(expr, parent_vars))

def myexec(expr):
    parent_vars = inspect.stack()[1][0].f_locals
    print(expr)
    exec(expr, parent_vars)

def test_long_tensor():
    PyTorch.manualSeed(123)
    print('test_long_tensor')
    a = PyTorch.LongTensor(3,2).geometric()
    myeval('a')
    myexec('a[1][1] = 9')
    myeval('a')
    myeval('a.size()')
    myeval('a + 2')
    myexec('a.resize2d(3,3).fill(1)')
    myeval('a')

