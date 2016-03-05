# {{header1}}
# {{header2}}

from lua cimport *

cdef extern from "nnWrapper.h":
    long pointerAsInt(void *ptr)
    void collectGarbage(lua_State *L)

{% for typedict in types %}
{% set Real = typedict['Real'] %}
{% set real = typedict['real'] %}
cdef extern from "nnWrapper.h":
    int TH{{Real}}Tensor_getRefCount(TH{{Real}}Tensor *self)
{% endfor %}

cdef extern from "nnWrapper.h":
    cdef struct lua_State
    lua_State *luaInit()
    void luaClose(lua_State *L)
    # void luaRequire(lua_State *L, const char *libName)

