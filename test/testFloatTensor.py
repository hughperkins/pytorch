from __future__ import print_function
import inspect
import PyTorch

def myeval(expr):
    parent_vars = inspect.stack()[1][0].f_locals
    print(expr, ':', eval(expr, parent_vars))

def myexec(expr):
    parent_vars = inspect.stack()[1][0].f_locals
    print(expr)
    exec(expr, parent_vars)

def test_float_tensor():
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

