from __future__ import print_function

import PyTorch
import inspect

def myeval(expr):
    parent_vars = inspect.stack()[1][0].f_locals
    print(expr, ':', eval(expr, parent_vars))

def test_long_tensor():
    print('test_long_tensor')
    a = PyTorch.LongTensor(3,2).geometric()
    myeval('a')

