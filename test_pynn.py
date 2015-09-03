from __future__ import print_function
import PyTorch

lua = PyTorch.getGlobalState().getLua()

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

class Linear(object):
    def __init__(self, lua, numIn, numOut):
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
        pushObject(lua, self)
        lua.getField(-1, name)
        pushGlobal(lua, 'torch', 'type')
        lua.insert(-2)
        lua.call(1, 1)
        typename = popString(lua)
#        print('attr typename', typename)
        pushObject(lua, self)
        lua.getField(-1, name)
        if typename == 'torch.FloatTensor':
            res = PyTorch._popFloatTensor()
            return res
        elif typename == 'function':
            def mymethod(*args):
                print('mymethod called method=', name)
                pushObject(lua, self)
                lua.getField(-1, name)
                pushObject(lua, self)
                for arg in args:
                    print('arg', arg, type(arg))
                    if isinstance(arg, PyTorch._FloatTensor):
                        print('arg is floattensor')
                        PyTorch._pushFloatTensor(arg)
                    else:
                        raise Exception('arg type ' + str(type(arg)) + ' not implemented')
                lua.call(len(args) + 1, 1)   # +1 for self
                lua.pushValue(-1)
                pushGlobal(lua, 'torch', 'type')
                lua.insert(-2)
                lua.call(1, 1)
                returntype = popString(lua)
                print('returntype', returntype)
                # this is getting a bit recursive :-P
                if returntype == 'torch.FloatTensor':
                    res = PyTorch._popFloatTensor()
                    return res
                else:
                    raise Exception('return type ' + str(returntype) + ' not implemented')
            return mymethod
        else:
            raise Exception('handling type ' + typename + ' not implemented')

linear = Linear(lua, 3, 5)
print('linear', linear)
print('linear.weight', linear.weight)
print('linear.output', linear.output)
print('linear.gradInput', linear.gradInput)

input = PyTorch.FloatTensor(4, 3).uniform()
print('input', input)
output = linear.updateOutput(input)
print('output', output)

gradInput = linear.updateGradInput(input, output)
print('gradInput', gradInput)

