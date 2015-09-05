# {{header1}}
# {{header2}}

{% set types = {'Long': 'long', 'Float': 'float', 'Double': 'double', 'Byte': 'unsigned char'} %}

cdef extern from "THRandom.h":
    cdef struct THGenerator

cdef extern from "nnWrapper.h":
    cdef struct lua_State

#cdef struct lua_State
#cdef struct THGenerator

{% for Real in types %}
{% set real = types[Real] %}

cdef extern from "THTensor.h":
    cdef struct TH{{Real}}Tensor

cdef class _{{Real}}Tensor(object):
    cdef TH{{Real}}Tensor *th{{Real}}Tensor
    cpdef int dims(self)
    cpdef set1d(self, int x0, {{real}} value)
    cpdef set2d(self, int x0, int x1, {{real}} value)
    cpdef {{real}} get1d(self, int x0)
    cpdef {{real}} get2d(self, int x0, int x1)
#    @cython.staticmethod
#    cdef fromNative(TH{{Real}}Tensor *tensorC, retain=*)
{% endfor %}

cdef class GlobalState:
#    cdef PyTorchState *state
    cdef lua_State *L
    cdef THGenerator *generator

