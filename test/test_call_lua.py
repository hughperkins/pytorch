import PyTorch
import PyTorchHelpers
import numpy as np


def test_call_lua():
    TestCallLua = PyTorchHelpers.load_lua_class('test/test_call_lua.lua', 'TestCallLua')

    batchSize = 2
    numFrames = 4
    inSize = 3
    outSize = 3
    kernelSize = 3

    luabit = TestCallLua('green')
    print(luabit.getName())
    assert luabit.getName() == 'green'
    print('type(luabit)', type(luabit))
    assert str(type(luabit)) == '<class \'PyTorchLua.TestCallLua\'>'

    np.random.seed(123)
    inTensor = np.random.randn(batchSize, numFrames, inSize).astype('float32')
    luain = PyTorch.asFloatTensor(inTensor)

    luaout = luabit.getOut(luain, outSize, kernelSize)

    outTensor = luaout.asNumpyTensor()
    print('outTensor', outTensor)
    # I guess we just assume if we got to this point, with no exceptions, then thats a good thing...
    # lets add some new test...

    outTensor = luabit.addThree(luain).asNumpyTensor()
    assert isinstance(outTensor, np.ndarray)
    assert inTensor.shape == outTensor.shape
    assert np.abs((inTensor + 3) - outTensor).max() < 1e-4

    res = luabit.printTable({'color': 'red', 'weather': 'sunny', 'anumber': 10, 'afloat': 1.234}, 'mistletoe', {
        'row1': 'col1', 'meta': 'data'})
    print('res', res)
    assert res == {'foo': 'bar', 'result': 12.345, 'bear': 'happy'}

    reslist = luabit.getList(['blue', 42, r'\omega'])
    restuple = luabit.getList((3.1415, r'~Python~', 'Torch', 1.2e-4))
    assert reslist == restuple == {1: 'lorem', 2: 'ipsum', 3: 42, 4: 2.1}