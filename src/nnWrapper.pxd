# GENERATED FILE, do not edit by hand
# Source: src/nnWrapper.jinja2.pxd



from lua cimport *

cdef extern from "nnWrapper.h":
    long pointerAsInt(void *ptr)
    void collectGarbage(lua_State *L)




cdef extern from "nnWrapper.h":
    int THLongTensor_getRefCount(THLongTensor *self)



cdef extern from "nnWrapper.h":
    int THFloatTensor_getRefCount(THFloatTensor *self)



cdef extern from "nnWrapper.h":
    int THDoubleTensor_getRefCount(THDoubleTensor *self)



cdef extern from "nnWrapper.h":
    int THByteTensor_getRefCount(THByteTensor *self)


cdef extern from "nnWrapper.h":
    cdef struct lua_State
    lua_State *luaInit()
    void luaClose(lua_State *L)
    # void luaRequire(lua_State *L, const char *libName)
