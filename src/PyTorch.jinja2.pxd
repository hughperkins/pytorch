# {{header1}}
# {{header2}}

cdef extern from "THRandom.h":
    cdef struct THGenerator

cdef extern from "nnWrapper.h":
    cdef struct lua_State

#cdef struct lua_State
#cdef struct THGenerator

{% for typedict in types %}
{% set Real = typedict['Real'] %}
{% set real = typedict['real'] %}

cdef extern from "THTensor.h":
    cdef struct TH{{Real}}Tensor

cdef class _{{Real}}Tensor(object):
    cdef object nparray
    cdef TH{{Real}}Tensor *native
    cdef {{real}} *data(self)
    # cpdef _{{Real}}Tensor contiguous(self)
    cpdef int dims(self)
    cpdef set1d(self, int x0, {{real}} value)
    cpdef set2d(self, int x0, int x1, {{real}} value)
    cpdef {{real}} get1d(self, int x0)
    cpdef {{real}} get2d(self, int x0, int x1)
    cpdef int isContiguous(self)
    cpdef {{real}} max(self)
    cpdef {{real}} min(self)

#    @cython.staticmethod
#    cdef fromNative(TH{{Real}}Tensor *tensorC, retain=*)
{% endfor %}

cdef class GlobalState:
#    cdef PyTorchState *state
    cdef lua_State *L
    cdef THGenerator *generator

cdef extern from "LuaHelper.h":
    void luaRequire(lua_State *L, const char *libName)
