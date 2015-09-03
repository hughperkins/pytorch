from __future__ import print_function

class Linear(object):
    def __init__(self):
        pass

    def __getattr__(self, name):
        print('Linear.__attr__', name)
        if name in ['foo', 'bar']:
            return 123
        else:
            def __method__(*args):
                print('static __method__', name)
                for arg in args:
                    print('   ', arg)

            return __method__

    def __setattr__(self, name, value):
        print('Linear.__setattr__', name, value)

    def __dir__(self):
        print('Linear.__dir__')
        return ['foo', 'bar']
    
    def __call__(self):
        print('Linear.__call__')

linear = Linear()
print('linear.foo', linear.foo)
linear.bar = 125
print(dir(linear))

linear()

linear.updateOutput('i')
linear.updateGradInput('i', 'go')

import Lua
lua = Lua.LuaState()
lua.pushNumber(35)
print(lua.toNumber(-1))

lua.getGlobal('require')
lua.pushString('torch')
lua.call(1, 0)
print('required torch')

lua.getGlobal('require')
lua.pushString('nn')
lua.call(1, 0)
print('required nn')

lua.getGlobal('require')
lua.pushString('cltorch')
lua.call(1, 0)
print('required cltorch')

lua.getGlobal('require')
lua.pushString('clnn')
lua.call(1, 0)

lua.getGlobal('torch')
lua.getField(-1, 'Tensor')
lua.pushNumber(2)
lua.pushNumber(3)
lua.call(2, 1)
lua.setGlobal("mytensor")

lua.getGlobal('torch')
lua.getField(-1, 'ClTensor')
lua.pushNumber(2)
lua.pushNumber(3)
lua.call(2, 1)
lua.setGlobal("mycltensor")

lua.getGlobal('torch')
lua.getField(-1, 'ClTensor')
lua.getField(-1, 'cl')
lua.getGlobal("mytensor")
lua.call(1, 1)

lua.getGlobal('torch')
lua.getField(-1, 'Tensor')
lua.getField(-1, '__tostring')
lua.getGlobal("mytensor")
lua.call(1, 1)
print(lua.toString(-1)) 

lua.getGlobal('torch')
lua.getField(-1, 'ClTensor')
lua.getField(-1, '__tostring')
lua.getGlobal("mycltensor")
lua.call(1, 1)
print(lua.toString(-1)) 

#def pushGlobal2(lua, name1, name2):
#    lua.getGlobal(name1)
#    lua.getField(-1, name2)
#    lua.remove(-2)

def pushGlobal(lua, name1, name2=None, name3=None):
    lua.getGlobal(name1)
    if name2 is None:
        return
    lua.getField(-1, name2)
    lua.remove(-2)
    if name3 is None:
        return
    lua.getField(-1, name3)
    lua.remove(-2)

def popString(lua):
    res = lua.toString(-1)
    lua.remove(-1)
    return res

class Linear(object):
    def __init__(self, numIn, numOut):
        print('Linear.__init__')
        pushGlobal(lua, 'nn', 'Linear')
        lua.pushNumber(numIn)
        lua.pushNumber(numOut)
        lua.call(2, 1)
        registerObject(lua, self)

        pushGlobal(lua, 'nn', 'Linear', 'float')
        pushObject(lua, self)
#        lua.getField(-1, 'float')
        lua.call(1, 0)

    def __del__(self):
        print('Linear.__del__')

    def __repr__(self):
        name = self.__class__.__name__
        pushGlobal(lua, 'nn', name, '__tostring')
        pushObject(lua, self)
        lua.call(1, 1)
        return popString(lua)

    def __getattr__(self, name):
        print('__getattr__', name)
        pushObject(lua, self)
        lua.getField(-1, 'weight')
        pushGlobal(lua, 'torch', 'type')
        lua.insert(-2)
        lua.call(1, 1)
        typename = popString(lua)
        print('attr typename', typename)

#linear = Linear()
#print('linear', linear)
#print('linear id', id(linear))
#print('linear id', id(linear))

#linear.foo = 'blah'
#l2 = Linear()
#l2.foo = 'paris'
#id1 = id(linear)
#id2 = id(l2)
#import ctypes
#probe = ctypes.cast(id1, ctypes.py_object).value
#print(probe.foo)
#probe = ctypes.cast(id2, ctypes.py_object).value
#print(probe.foo)

def registerObject(lua, myobject):
    lua.pushNumber(id(myobject))
    lua.insert(-2)
    lua.setRegistry()

#    pushObject(lua, myobject)
#    lua.pushNumber(id(myobject))
#    lua.setRegistry()

def unregisterObject(lua, myobject):
#    pushObject(lua, myobject)
#    lua.pushNil()
#    lua.setRegistry()

    lua.pushNumber(id(myobject))
    lua.pushNil()
    lua.setRegistry()

def pushObject(lua, myobject):
    lua.pushNumber(id(myobject))
    lua.getRegistry()

linear = Linear(3,4)
print('linear', linear)
print('linear.weight', linear.weight)

#lua.getGlobal('nn')
#lua.getField(-1, 'Linear')
#lua.pushNumber(3)
#lua.pushNumber(4)
#lua.call(2, 1)
#registerObject(lua, linear)

#lua.getGlobal('nn')
#lua.getField(-1, 'Linear')
#lua.getField(-1, '__tostring')
#pushObject(lua, linear)
#lua.call(1, 1)
#print(lua.toString(-1))

#    lua_pushlightuserdata(L, instanceKey);
#    lua_insert(L, -2);
#    lua_settable(L, LUA_REGISTRYINDEX);

#    lua_pushlightuserdata(L, instanceKey);
#    lua_pushnil(L);
#    lua_settable(L, LUA_REGISTRYINDEX);

print('linear', linear)

#import ctypes
#myid = id(linear)
#linear = None
#print('getting probe...')
#beforevalue = ctypes.cast(myid, ctypes.py_object)
#print('type(beforevalue)', type(beforevalue))
#probe = beforevalue.value
#print('...got probe')
#print('probe is None', probe is None)
#print(probe)


