"""
Created to address https://github.com/hughperkins/pytorch/issues/10
"""
from __future__ import print_function
import PyTorchHelpers
import numpy as np
import sys
import gc


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

    # v = list(range(10))
    # print('v ref', sys.getrefcount(v))

    print('\nallocating npv')
    npv = np.asarray(list(range(10)), dtype=np.float32)
    # print('v ref', sys.getrefcount(v))
    print('npv ref', sys.getrefcount(npv))

    gc.collect()
    print('\nafter gc')
    print('npv ref', sys.getrefcount(npv))
    print('npv.data ref', sys.getrefcount(npv.data))

    npv_data = npv.data
    print('\nafter get np.vdata')
    # print('v ref', sys.getrefcount(v))
    print('npv ref', sys.getrefcount(npv))
    print('npv_data ref', sys.getrefcount(npv_data))
    print('type(npv_data)', type(npv_data))
    # npv_data_str = 'npv_data '
    # for value in npv_data:
    #     npv_data_str += ' ' + int(value)
    # print(npv_data_str)
    # print('npv_data', npv_data)

    npv2 = npv * 2
    print('\nafter assign npv2')
    # print('v ref', sys.getrefcount(v))
    print('npv ref', sys.getrefcount(npv))
    print('npv2 ref', sys.getrefcount(npv2))
    print('npv_data ref', sys.getrefcount(npv_data))
    print('npv2.data ref', sys.getrefcount(npv2.data))

    npv2 = None
    gc.collect()
    print('\nafter set npv2 None, and gc')
    # print('v ref', sys.getrefcount(v))
    print('npv ref', sys.getrefcount(npv))
    print('npv_data ref', sys.getrefcount(npv_data))

    print('\ndoing lua_obj.set(npv)')
    lua_obj.set(npv)
    # print('v ref', sys.getrefcount(v))
    print('npv ref', sys.getrefcount(npv))
    print('npv_data ref', sys.getrefcount(npv_data))

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
