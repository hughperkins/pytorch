"""
Created to address https://github.com/hughperkins/pytorch/issues/10
"""
from __future__ import print_function
import PyTorchHelpers
import numpy as np


def test_inline_refcount_test1():
    TestInlineRefCount = PyTorchHelpers.load_lua_class('test/test_inline_refcount.lua', 'TestInlineRefCount')
    lua_obj = TestInlineRefCount()

    v = list(range(10))
    npv = np.asarray(v, dtype=np.float32)

    lua_obj.set(npv)
    print('lua_obj.v.asNumpyTensor()', lua_obj.v.asNumpyTensor())
    assert np.abs(lua_obj.v.asNumpyTensor() - npv).max() <= 1e-4

    npv2 = npv * 2
    lua_obj.set(npv2)
    print('lua_obj.v.asNumpyTensor()', lua_obj.v.asNumpyTensor())
    assert np.abs(lua_obj.v.asNumpyTensor() - npv).max() <= 1e-4


def test_inline_refcount_test2():
    TestInlineRefCount = PyTorchHelpers.load_lua_class('test/test_inline_refcount.lua', 'TestInlineRefCount')
    lua_obj = TestInlineRefCount()

    v = list(range(10))
    npv = np.asarray(v, dtype=np.float32)

    lua_obj.set(npv)
    assert lua_obj.v[1] != 0
    assert npv[1] != 0

    npv2 = npv * 2
    lua_obj.set(npv2)
    assert lua_obj.v[1] != 0
    assert npv[1] != 0

    lua_obj.set(npv * 3)
    assert lua_obj.v[1] != 0
    assert npv[1] != 0

    assert npv[1] != 0
    assert lua_obj.v[1] != 0

    print('allocate simple array of zeros, this overwrites lua_obj.v :-P')
    np.zeros(10, dtype=np.float32)

    print('')
    assert npv[1] != 0
    assert lua_obj.v[1] != 0
    assert npv[1] != 0
