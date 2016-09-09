from __future__ import print_function, division
import PyTorch
import PyTorchAug
import numpy as np
# from test.test_helpers import myeval


def test_save_load():
    np.random.seed(123)
    a_np = np.random.randn(3, 2).astype(np.float32)
    a = PyTorch.asFloatTensor(a_np)
    print('a', a)

    filename = '/tmp/foo.t7'  # TODO: should use tempfile to get this
    PyTorchAug.save(filename, a)

    b = PyTorchAug.load(filename)
    print('type(b)', type(b))
    print('b', b)

    assert np.abs(a_np - b.asNumpyTensor()).max() < 1e-4
