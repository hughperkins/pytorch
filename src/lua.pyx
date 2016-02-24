from lua cimport *

# GENERATED FILE, do not edit by hand
# Source: src/lua.jinja2.pyx



#import PyTorch
#cimport PyTorch
from nnWrapper cimport *

# from PyTorch cimport *

LUA_REGISTRYINDEX = getLuaRegistryIndex()

#cdef PyTorch.GlobalState globalState = PyTorch.getGlobalState()

#cdef class LuaHelper(object):
#    @staticmethod
#    def require(name):
#        require(globalState.L, name)

cdef class LuaState(object):
    # property in .pxd file

    def __cinit__(self, _allocate=True):
#        print('LuaState.__cinit__')
        if _allocate:
            self.L = luaInit()

    def __dealloc__(self):
#        print('LuaState.__dealloc__')
        pass

    def insert(self, int index):
        lua_insert(self.L, index)

    def remove(self, int index):
        lua_remove(self.L, index)

    def pushNumber(self, float number):
        lua_pushnumber(self.L, number)

    def pushString(self, mystring):
        lua_pushstring(self.L, mystring.encode('utf-8'))

    def toString(self, int index):
        cdef bytes py_string = lua_tostring(self.L, index)
        return py_string

    def toNumber(self, int index):
        return lua_tonumber(self.L, index)

    def getGlobal(self, name):
        lua_getglobal(self.L, name.encode('utf-8'))

    def setGlobal(self, name):
        lua_setglobal(self.L, name.encode('utf-8'))

    def pushNil(self):
        lua_pushnil(self.L)

    def pushValue(self, int index):
        lua_pushvalue(self.L, index)

    def call(self, int numIn, int numOut):
        lua_call(self.L, numIn, numOut)

    def getField(self, int index, name):
        lua_getfield(self.L, index, name.encode('utf-8'))

    def setRegistry(self):
        lua_settable(self.L, LUA_REGISTRYINDEX)

    def getRegistry(self):
        lua_gettable(self.L, LUA_REGISTRYINDEX)

    def next(self, int index):
        return lua_next(self.L, index)

    def getTop(self):
        return lua_gettop(self.L)

    def isUserData(self, int index):
        return lua_isuserdata(self.L, index)

cdef LuaState_fromNative(lua_State *L):
    cdef LuaState luaState = LuaState(_allocate=False)
    luaState.L = L
    return luaState
