import PyTorch
import PyTorchAug

def load_lua_class(lua_filename, lua_classname):
    module = lua_filename.replace('.lua', '')
    PyTorch.require(module)
    class LuaWrapper(PyTorchAug.LuaClass):
        def __init__(self, _fromLua=False):
            self.luaclass = lua_classname
            if not _fromLua:
                name = lua_classname
                super(self.__class__, self).__init__([name])
            else:
                self.__dict__['__objectId'] = getNextObjectId()
    return LuaWrapper

