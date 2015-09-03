cdef extern from "lua.h":
    struct lua_State
    int LUA_REGISTRYINDEX
    void lua_pushnumber(lua_State *L, float number)
    float lua_tonumber(lua_State *L, int index)
    void lua_pushstring(lua_State *L, const char *value)
    const char *lua_tostring(lua_State *L, int index)
    void lua_call(lua_State *L, int argsIn, int argsOut)
    void lua_remove(lua_State *L, int index)
    void lua_insert(lua_State *L, int index)
    void lua_getglobal(lua_State *L, const char *name)
    void lua_setglobal(lua_State *L, const char *name)
    void lua_settable(lua_State *L, int index)
    void lua_gettable(lua_State *L, int index)
    void lua_getfield(lua_State *L, int index, const char *name)
    void lua_pushnil(lua_State *L)

cdef extern from "LuaHelper.h":
    lua_State *luaInit()

cdef class LuaState(object):
    cdef lua_State *L

    def __cinit__(self):
        print('LuaState.__cinit__')
        self.L = luaInit()

    def __dealloc__(self):
        print('LuaState.__dealloc__')
        pass

    def insert(self, int index):
        lua_insert(self.L, index)

    def remove(self, int index):
        lua_remove(self.L, index)

    def pushNumber(self, float number):
        lua_pushnumber(self.L, number)

    def pushString(self, mystring):
        lua_pushstring(self.L, mystring)

    def toString(self, int index):
        cdef bytes py_string = lua_tostring(self.L, index)
        return py_string

    def toNumber(self, int index):
        return lua_tonumber(self.L, index)

    def getGlobal(self, name):
        lua_getglobal(self.L, name)

    def setGlobal(self, name):
        lua_setglobal(self.L, name)

    def pushNil(self):
        lua_pushnil(self.L)

    def call(self, int numIn, int numOut):
        lua_call(self.L, numIn, numOut)

    def getField(self, int index, name):
        lua_getfield(self.L, index, name)

    def setRegistry(self):
        lua_settable(self.L, LUA_REGISTRYINDEX)

    def getRegistry(self):
        lua_gettable(self.L, LUA_REGISTRYINDEX)

cdef LuaState_fromNative(lua_State *L):
    cdef LuaState luaState = LuaState()
    luaState.L = L
    return luaState

#def init():
#    global

#init()

