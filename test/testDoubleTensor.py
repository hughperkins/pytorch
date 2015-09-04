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

def test_double_tensor():
    LongTensor = PyTorch.LongTensor
    DoubleTensor = PyTorch.DoubleTensor
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
    myexec('a.resize2d(3,3)')
    myeval('a')
    myexec('size = LongTensor(2)')
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

