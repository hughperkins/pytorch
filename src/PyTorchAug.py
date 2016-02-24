from __future__ import print_function
import PyTorch

lua = PyTorch.getGlobalState().getLua()

nextObjectId = 1
def getNextObjectId():
    global nextObjectId
    res = nextObjectId
    nextObjectId += 1
    return res

class Linear(object):
    def __init__(self):
        print('Linear.__init__')

    def __attr__(self):
        print('Linear.__attr__')

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

def pushGlobalFromList(lua, nameList):
    lua.getGlobal(nameList[0])
    for name in nameList[1:]:
        lua.getField(-1, name)
        lua.remove(-2)

def popString(lua):
    res = lua.toString(-1)
    lua.remove(-1)
    return res.decode('utf-8')

def registerObject(lua, myobject):
    lua.pushNumber(myobject.__objectId)
    lua.insert(-2)
    lua.setRegistry()

#    pushObject(lua, myobject)
#    lua.pushNumber(id(myobject))
#    lua.setRegistry()

def unregisterObject(lua, myobject):
#    pushObject(lua, myobject)
#    lua.pushNil()
#    lua.setRegistry()

    lua.pushNumber(myobject.__objectId)
    lua.pushNil()
    lua.setRegistry()

def pushObject(lua, myobject):
    lua.pushNumber(myobject.__objectId)
    lua.getRegistry()

def torchType(lua, pos):
    lua.pushValue(-1)
    pushGlobal(lua, "torch", "type")
    lua.insert(-2)
    lua.call(1, 1)
    return popString(lua)

class LuaClass(object):
    def __init__(self, nameList, *args):
        # print('LuaClass.__init__()')
        lua = PyTorch.getGlobalState().getLua()
#        self.luaclass = luaclass
        self.__dict__['__objectId'] = getNextObjectId()
        topStart = lua.getTop()
        pushGlobalFromList(lua, nameList)
        for arg in args:
            if isinstance(arg, int):
                lua.pushNumber(arg)
            else:
                raise Exception('arg type ' + str(type(arg)) + ' not implemented')
        lua.call(len(args), 1)
        registerObject(lua, self)

#        nameList = nameList[:]
#        nameList.append('float')
#        pushGlobalFromList(lua, nameList)
#        pushObject(lua, self)
#        lua.call(1, 0)

        topEnd = lua.getTop()
        assert topStart == topEnd

    def __del__(self):
        name = self.__class__.__name__
#        print(name + '.__del__')

    def __repr__(self):
        topStart = lua.getTop()
#        name = self.__class__.__name__
        luaClass = self.luaclass
        if luaClass == 'table':
            return 'table'
        splitLuaClass = luaClass.split('.')
        if len(splitLuaClass) == 1:
            pushGlobal(lua, splitLuaClass[0], '__tostring')
        elif len(splitLuaClass) == 2:
            pushGlobal(lua, splitLuaClass[0], splitLuaClass[1], '__tostring')
        else:
            raise Exception('not implemented: luaclass with more than 2 parts ' + luaClass)
#        pushGlobal(lua, 'nn', name, '__tostring')
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
        if name == '__objectId':
            return self.__dict__['__objectId']
        topStart = lua.getTop()
        pushObject(lua, self)
        lua.getField(-1, name)
        lua.remove(-2)
        pushGlobal(lua, 'torch', 'type')
        lua.insert(-2)
        lua.call(1, 1)
        typename = popString(lua)
        pushObject(lua, self)
        lua.getField(-1, name)
        lua.remove(-2)
        if typename in cythonClasses:
            popFunction = cythonClasses[typename]['popFunction']
            res = popFunction()
            topEnd = lua.getTop()
            assert topStart == topEnd
            return res
        elif typename == 'function':
            def mymethod(*args):
                topStart = lua.getTop()
                pushObject(lua, self)
                lua.getField(-1, name)
                lua.insert(-2)
#                pushObject(lua, self)
                for arg in args:
#                    print('arg', arg, type(arg))
                    pushedArg = False
                    for pythonClass in pushFunctionByPythonClass:
                        if isinstance(arg, pythonClass):
                            pushFunctionByPythonClass[pythonClass](arg)
                            pushedArg = True
                            break
                    if not pushedArg and type(arg) in luaClassesReverse:
                        pushObject(lua, arg)
                        pushedArg = True
                    if not pushedArg and isinstance(arg, float):
                        lua.pushNumber(arg)
                        pushedArg = True
                    if not pushedArg and isinstance(arg, int):
                        lua.pushNumber(arg)
                        pushedArg = True
                    if not pushedArg and isinstance(arg, str):
                        lua.pushString(arg)
                        pushedArg = True
                    if not pushedArg:
                        raise Exception('arg type ' + str(type(arg)) + ' not implemented')
                lua.call(len(args) + 1, 1)   # +1 for self
                lua.pushValue(-1)
                pushGlobal(lua, 'torch', 'type')
                lua.insert(-2)
                lua.call(1, 1)
                returntype = popString(lua)
                # this is getting a bit recursive :-P
#                print('cythonClasses', cythonClasses)
                if returntype in cythonClasses:
                    popFunction = cythonClasses[returntype]['popFunction']
                    res = popFunction()
                    topEnd = lua.getTop()
                    assert topStart == topEnd
                    return res
                elif returntype == 'number':
                    res = lua.toNumber(-1)
                    lua.remove(-1)
                    topEnd = lua.getTop()
                    assert topStart == topEnd
                    return res
                elif returntype in luaClasses:
                    returnobject = luaClasses[returntype](_fromLua=True)
                    registerObject(lua, returnobject)
                    topEnd = lua.getTop()
                    assert topStart == topEnd
                    return returnobject
                elif returntype == 'nil':
                    lua.remove(-1)
                    topEnd = lua.getTop()
                    assert topStart == topEnd
                    return None
                else:
                    raise Exception('return type ' + str(returntype) + ' not implemented')
            lua.remove(-1)
            topEnd = lua.getTop()
            assert topStart == topEnd
            return mymethod
        elif typename == 'nil':
            lua.remove(-1)
            topEnd = lua.getTop()
            assert topStart == topEnd
            return None
        else:
            raise Exception('handling type ' + typename + ' not implemented')

class Table(LuaClass):
    def __init__(self, _fromLua=False):
        # print('Table.__init__')
        if not _fromLua:
            name = self.__class__.__name__
            super(self.__class__, self).__init__(['nn', name])
        else:
            self.__dict__['__objectId'] = getNextObjectId()
            self.luaclass = 'table'

class Linear(LuaClass):
    def __init__(self, numIn=1, numOut=1, _fromLua=False):
        # print('Linear.__init__')
        self.luaclass = 'nn.Linear'
        if not _fromLua:
            name = self.__class__.__name__
            super(self.__class__, self).__init__(['nn', name], numIn, numOut)
        else:
            self.__dict__['__objectId'] = getNextObjectId()

class ClassNLLCriterion(LuaClass):
    def __init__(self, _fromLua=False):
        self.luaclass = 'nn.ClassNLLCriterion'
        if not _fromLua:
            name = self.__class__.__name__
            super(self.__class__, self).__init__(['nn', name])
        else:
            self.__dict__['__objectId'] = getNextObjectId()

class MSECriterion(LuaClass):
    def __init__(self, _fromLua=False):
        self.luaclass = 'nn.MSECriterion'
        if not _fromLua:
            name = self.__class__.__name__
            super(self.__class__, self).__init__(['nn', name])
        else:
            self.__dict__['__objectId'] = getNextObjectId()

class Sequential(LuaClass):
    def __init__(self, _fromLua=False):
        self.luaclass = 'nn.Sequential'
        if not _fromLua:
            name = self.__class__.__name__
            super(self.__class__, self).__init__(['nn', name])
        else:
            self.__dict__['__objectId'] = getNextObjectId()

class LogSoftMax(LuaClass):
    def __init__(self, _fromLua=False):
        self.luaclass = 'nn.LogSoftMax'
        if not _fromLua:
            name = self.__class__.__name__
            super(self.__class__, self).__init__(['nn', name])
        else:
            self.__dict__['__objectId'] = getNextObjectId()

class Reshape(LuaClass):
    def __init__(self, s1, s2=None, s3=None, s4=None, _fromLua=False):
        self.luaclass = 'nn.Reshape'
        if not _fromLua:
            name = self.__class__.__name__
            if s4 is not None:   # this is a bit hacky, but gets it working for now...
                super(self.__class__, self).__init__(['nn', name], s1, s2, s3, s4)
            elif s3 is not None:
                super(self.__class__, self).__init__(['nn', name], s1, s2, s3)
            elif s2 is not None:
                super(self.__class__, self).__init__(['nn', name], s1, s2)
            else:
                super(self.__class__, self).__init__(['nn', name], s1)
        else:
            self.__dict__['__objectId'] = getNextObjectId()

class SpatialConvolutionMM(LuaClass):
    def __init__(self, nInputPlane, nOutputPlane, kW, kH, dW=1, dH=1, padW=0, padH=0, _fromLua=False):
        self.luaclass = 'nn.SpatialConvolutionMM'
        if not _fromLua:
            name = self.__class__.__name__
            super(self.__class__, self).__init__(['nn', name], nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
        else:
            self.__dict__['__objectId'] = getNextObjectId()

class SpatialMaxPooling(LuaClass):
    def __init__(self, kW, kH, dW, dH, padW=0, padH=0, _fromLua=False):
        self.luaclass = 'nn.SpatialMaxPooling'
        if not _fromLua:
            name = self.__class__.__name__
            super(self.__class__, self).__init__(['nn', name], kW, kH, dW, dH, padW, padH)
        else:
            self.__dict__['__objectId'] = getNextObjectId()

class ReLU(LuaClass):
    def __init__(self, _fromLua=False):
        self.luaclass = 'nn.ReLU'
        if not _fromLua:
            name = self.__class__.__name__
            super(self.__class__, self).__init__(['nn', name])
        else:
            self.__dict__['__objectId'] = getNextObjectId()

class Tanh(LuaClass):
    def __init__(self, _fromLua=False):
        self.luaclass = 'nn.Tanh'
        if not _fromLua:
            name = self.__class__.__name__
            super(self.__class__, self).__init__(['nn', name])
        else:
            self.__dict__['__objectId'] = getNextObjectId()

luaClasses = {}
luaClasses['nn.Reshape'] = Reshape
luaClasses['nn.Linear'] = Linear
luaClasses['nn.ClassNLLCriterion'] = ClassNLLCriterion
luaClasses['nn.MSECriterion'] = MSECriterion
luaClasses['nn.Sequential'] = Sequential
luaClasses['nn.LogSoftMax'] = LogSoftMax
luaClasses['nn.SpatialConvolutionMM'] = SpatialConvolutionMM
luaClasses['nn.SpatialMaxPooling'] = SpatialMaxPooling
luaClasses['nn.ReLU'] = ReLU
luaClasses['nn.Tanh'] = Tanh
luaClasses['table'] = Table

luaClassesReverse = {}
def populateLuaClassesReverse():
    luaClassesReverse.clear()
    for name in luaClasses:
        classtype = luaClasses[name]
        luaClassesReverse[classtype] = name
populateLuaClassesReverse()

cythonClasses = {}
cythonClasses['torch.FloatTensor'] = {'popFunction': PyTorch._popFloatTensor}
cythonClasses['torch.DoubleTensor'] = {'popFunction': PyTorch._popDoubleTensor}

pushFunctionByPythonClass = {}
pushFunctionByPythonClass[PyTorch._FloatTensor] = PyTorch._pushFloatTensor
pushFunctionByPythonClass[PyTorch._DoubleTensor] = PyTorch._pushDoubleTensor

