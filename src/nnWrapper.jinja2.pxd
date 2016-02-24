# {{header1}}
# {{header2}}

{% set types = {
    'Long': {'real': 'long'},
    'Float': {'real': 'float'}, 
    'Double': {'real': 'double'},
    'Byte': {'real': 'unsigned char'}
}
%}

from lua cimport *

cdef extern from "nnWrapper.h":
    long pointerAsInt(void *ptr)
    void collectGarbage(lua_State *L)

{% for Real in types %}
cdef extern from "nnWrapper.h":
    int TH{{Real}}Tensor_getRefCount(TH{{Real}}Tensor *self)
{% endfor %}

cdef extern from "nnWrapper.h":
    cdef struct lua_State
    lua_State *luaInit()
    void luaClose(lua_State *L)
    # void luaRequire(lua_State *L, const char *libName)

