from __future__ import print_function
import PyTorch
import PyTorchLua
import PyTorchHelpers

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

def unregisterObject(lua, myobject):
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

def pushSomething(lua, something):
    if isinstance(something, int):
        lua.pushNumber(something)
        return

    if isinstance(something, float):
        lua.pushNumber(something)
        return

    if isinstance(something, str):
        lua.pushString(something)
        return

    if isinstance(something, dict):
        pushTable(lua, something)
        return

    for pythonClass in pushFunctionByPythonClass:
        if isinstance(something, pythonClass):
            pushFunctionByPythonClass[pythonClass](something)
            return

    if type(something) in luaClassesReverse:
        pushObject(lua, something)
        return

    raise Exception('pushing type ' + str(type(something)) + ' not implemented, value ', something)

def popSomething(lua, self=None, name=None):
    lua.pushValue(-1)
    pushGlobal(lua, 'torch', 'type')
    lua.insert(-2)
    lua.call(1, 1)
    typestring = popString(lua)

    if typestring in cythonClasses:
        popFunction = cythonClasses[typestring]['popFunction']
        res = popFunction()
        return res

    if typestring == 'number':
        res = lua.toNumber(-1)
        lua.remove(-1)
        return res

    if typestring == 'string':
        res = popString(lua)
        return res

    if typestring == 'table':
        return popTable(lua)

    if typestring in luaClasses:
        returnobject = luaClasses[typestring](_fromLua=True)
        registerObject(lua, returnobject)
        return returnobject

    if typestring == 'function':
        def mymethod(*args):
            topStart = lua.getTop()
            pushObject(lua, self)
            lua.getField(-1, name)
            lua.insert(-2)
            for arg in args:
                pushSomething(lua, arg)
            lua.call(len(args) + 1, 1)   # +1 for self
            res = popSomething(lua)
            topEnd = lua.getTop()
            assert topStart == topEnd
            return res
        lua.remove(-1)
        return mymethod

    if typestring == 'nil':
        lua.remove(-1)
        return None

    raise Exception('pop type ' + str(typestring) + ' not implemented')
    # print('pop type ' + str(typestring) + ' not implemented')

def pushTable(lua, table):
    lua.newTable()
    for k, v in table.items():
        pushSomething(lua, k)
        pushSomething(lua, v)
        lua.setTable(-3)

def popTable(lua):
    res = {}
    lua.pushNil()
    while lua.next(-2) != 0:
        value = popSomething(lua)
        lua.pushValue(-1)
        key = popSomething(lua)
        res[key] = value
    lua.remove(-1)
    return res


class LuaClass(object):
    def __init__(self, *args, nameList):
        lua = PyTorch.getGlobalState().getLua()
#        self.luaclass = luaclass
        self.__dict__['__objectId'] = getNextObjectId()
        topStart = lua.getTop()
        pushGlobalFromList(lua, nameList)
        for arg in args:
            pushSomething(lua, arg)
        lua.call(len(args), 1)
        registerObject(lua, self)

        topEnd = lua.getTop()
        assert topStart == topEnd

    def __del__(self):
        name = self.__class__.__name__

    def __repr__(self):
        topStart = lua.getTop()
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
        res = popSomething(lua, self, name)
        topEnd = lua.getTop()
        assert topStart == topEnd
        return res

def loadNnClass(nnClassName):
    class AnNnClass(LuaClass):
        def __init__(self, *args, _fromLua=False, **kwargs):
            self.luaclass = 'nn.' + nnClassName
            if not _fromLua:
                LuaClass.__init__(self, *args, nameList=['nn', nnClassName], **kwargs)
            else:
                self.__dict__['__objectId'] = getNextObjectId()
    renamedClass = type(AnNnClass)(nnClassName, (AnNnClass,), {})
    return renamedClass


luaClasses = {}
nnClasses = [
    'Linear', 'ClassNLLCriterion', 'MSECriterion', 'Sequential', 'LogSoftMax',
    'Reshape', 'SpatialConvolutionMM', 'SpatialMaxPooling', 'ReLU', 'Tanh']
for nnClassName in nnClasses:
    nnClass = loadNnClass(nnClassName)
    globals()[nnClassName] = nnClass
    luaClasses['nn.' + nnClassName] = nnClass


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

