import PyTorch
import PyTorchAug
import types
import PyTorchLua

def load_lua_class(lua_filename, lua_classname):
    module = lua_filename.replace('.lua', '')
    PyTorch.require(module)
    splitName = lua_classname.split('.')
    class LuaWrapper(PyTorchAug.LuaClass):
        def __init__(self, *args):
            _fromLua = False
            if len(args) >= 1:
                if args[0] == '__FROMLUA__':
                   _fromLua = True
                   args = args[1:]
#            print('LuaWrapper.__init__', lua_classname, 'fromLua', _fromLua, 'args', args)
            self.luaclass = lua_classname
            if not _fromLua:
                PyTorchAug.LuaClass.__init__(self, splitName, *args)
            else:
                self.__dict__['__objectId'] = PyTorchAug.getNextObjectId()
    renamedClass = PyTorchLua.renameClass(LuaWrapper, module, lua_classname)
    return renamedClass

