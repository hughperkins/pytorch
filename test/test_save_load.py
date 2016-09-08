from __future__ import print_function, division
import PyTorch
import PyTorchAug
import numpy as np
import inspect


def myeval(expr):
    parent_vars = inspect.stack()[1][0].f_locals
    print(expr, ':', eval(expr, parent_vars))


def myexec(expr):
    parent_vars = inspect.stack()[1][0].f_locals
    print(expr)
    exec(expr, parent_vars)


def test_save_load():
    np.random.seed(123)
    a_np = np.random.randn(3, 2).astype(np.float32)
    a = PyTorch.asFloatTensor(a_np)
    print('a', a)

    filename = '/tmp/foo.t7'  # TODO: should use tempfile to get this
    PyTorchAug.save(filename, a)
