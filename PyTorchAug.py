from __future__ import print_function
import PyTorch

lua = PyTorch.getGlobalState().getLua()

class Linear(object):
    def __init__(self):
        print('Linear.__init__')

    def __attr__(self):
        print('Linear.__attr__')

def pushGlobal(lua, name1, name2=None, name3=None):
    print('pushglobal START lua_gettop', lua.getTop())
    lua.getGlobal(name1)
    if name2 is None:
        print('pushglobal END lua_gettop', lua.getTop())
        return
    lua.getField(-1, name2)
    lua.remove(-2)
    if name3 is None:
        print('pushglobal END lua_gettop', lua.getTop())
        return
    lua.getField(-1, name3)
    lua.remove(-2)
    print('pushglobal END lua_gettop', lua.getTop())

def pushGlobalFromList(lua, nameList):
    print('pushglobal START lua_gettop', lua.getTop())
    lua.getGlobal(nameList[0])
    for name in nameList[1:]:
        lua.getField(-1, name)
        lua.remove(-2)
    print('pushglobal END lua_gettop', lua.getTop())

def popString(lua):
    res = lua.toString(-1)
    lua.remove(-1)
    return res

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

luaClasses = {}

class LuaClass(object):
    def __init__(self, nameList, *args):
        print('LuaClass.initNew', nameList)
        lua = PyTorch.getGlobalState().getLua()
        topStart = lua.getTop()
        pushGlobalFromList(lua, nameList)
        for arg in args:
            print('    arg=', type(arg))
            if isinstance(arg, int):
                lua.pushNumber(arg)
            else:
                raise Exception('arg type ' + str(type(arg)) + ' not implemented')
        lua.call(len(args), 1)
        registerObject(lua, self)
        topEnd = lua.getTop()
        assert topStart == topEnd

    def __del__(self):
        name = self.__class__.__name__
        print(name + '.__del__')

    def __repr__(self):
        topStart = lua.getTop()
        name = self.__class__.__name__
        pushGlobal(lua, 'nn', name, '__tostring')
        pushObject(lua, self)
        lua.call(1, 1)
        res = popString(lua)
        topEnd = lua.getTop()
        assert topStart == topEnd
        return res

    def __dir__(self):
        topStart = lua.getTop()
        attributes = []
        pushObject(lua, self)
        lua.pushNil()
        while(lua.next(-2)) != 0:
            keyname = lua.toString(-2)
            attributes.append(keyname)
            lua.remove(-1)
        lua.remove(-1)
        topEnd = lua.getTop()
        assert topStart == topEnd
        return attributes

    def __getattr__(self, name):
        print('getattr top', lua.getTop())
        topStart = lua.getTop()
        pushObject(lua, self)
        lua.getField(-1, name)
        lua.remove(-2)
        pushGlobal(lua, 'torch', 'type')
        lua.insert(-2)
        lua.call(1, 1)
        typename = popString(lua)
#        print('attr typename', typename)
        pushObject(lua, self)
        lua.getField(-1, name)
        lua.remove(-2)
        if typename == 'torch.FloatTensor':
            res = PyTorch._popFloatTensor()
            topEnd = lua.getTop()
            assert topStart == topEnd
            return res
        elif typename == 'function':
            print('getattr function top', lua.getTop())
            def mymethod(*args):
                topStart = lua.getTop()
                print('mymethod called method=', name)
                print('getattr mymethod ', lua.getTop())
                pushObject(lua, self)
                lua.getField(-1, name)
                lua.insert(-2)
#                pushObject(lua, self)
                for arg in args:
                    print('arg', arg, type(arg))
                    if isinstance(arg, PyTorch._FloatTensor):
                        print('arg is floattensor')
                        PyTorch._pushFloatTensor(arg)
                    elif type(arg) in luaClassesReverse:
                        print('found ' + str(type(arg)) + ' in luaClassesReverse')
                        pushObject(lua, arg)
                    else:
                        raise Exception('arg type ' + str(type(arg)) + ' not implemented')
                lua.call(len(args) + 1, 1)   # +1 for self
                lua.pushValue(-1)
                pushGlobal(lua, 'torch', 'type')
                lua.insert(-2)
                lua.call(1, 1)
                returntype = popString(lua)
                print('returntype', returntype)
                print('getattr mymethod after getting returntype ', lua.getTop())
                # this is getting a bit recursive :-P
                if returntype == 'torch.FloatTensor':
                    res = PyTorch._popFloatTensor()
                    topEnd = lua.getTop()
                    print('topstart', topStart, 'topend', topEnd)
                    print('topstart - topend', topStart - topEnd)
                    assert topStart == topEnd
                    return res
                elif returntype in luaClasses:
                    returnobject = luaClasses[returntype](_fromLua=True)
                    registerObject(lua, returnobject)
#                    lua.remove(-1)
                    topEnd = lua.getTop()
                    assert topStart == topEnd
                    return returnobject
                else:
                    raise Exception('return type ' + str(returntype) + ' not implemented')
            lua.remove(-1)
            topEnd = lua.getTop()
            assert topStart == topEnd
            return mymethod
        else:
            raise Exception('handling type ' + typename + ' not implemented')

class Linear(LuaClass):
    def __init__(self, numIn=1, numOut=1, _fromLua=False):
        if not _fromLua:
            name = self.__class__.__name__
            super(self.__class__, self).__init__(['nn', name], numIn, numOut)

class ClassNLLCriterion(LuaClass):
    def __init__(self, _fromLua=False):
        if not _fromLua:
            name = self.__class__.__name__
            super(self.__class__, self).__init__(['nn', name])

class Sequential(LuaClass):
    def __init__(self, _fromLua=False):
        if not _fromLua:
            name = self.__class__.__name__
            super(self.__class__, self).__init__(['nn', name])

luaClasses['nn.Linear'] = Linear
luaClasses['nn.ClassNLLCriterion'] = ClassNLLCriterion
luaClasses['nn.Sequential'] = Sequential

luaClassesReverse = {}
def populateLuaClassesReverse():
    for name in luaClasses:
        classtype = luaClasses[name]
        luaClassesReverse[classtype] = name
populateLuaClassesReverse()

