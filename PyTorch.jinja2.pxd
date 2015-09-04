# Original file is: PyTorch.jinja2.pxd
# If you are looking at PyTorch.pxd, it is a generated file, dont edit it directly ;-)

{% set types = ['Long', 'Float'] %}

cdef extern from "THRandom.h":
    cdef struct THGenerator

cdef extern from "nnWrapper.h":
    cdef struct lua_State

#cdef struct lua_State
#cdef struct THGenerator

{% for Real in types %}
cdef extern from "THTensor.h":
    cdef struct TH{{Real}}Tensor

cdef class _{{Real}}Tensor(object):
    cdef TH{{Real}}Tensor *th{{Real}}Tensor

{% if Real == 'Float' %}
    cpdef int dims(self)
    cpdef set1d(self, int x0, float value)
    cpdef set2d(self, int x0, int x1, float value)
    cpdef float get1d(self, int x0)
    cpdef float get2d(self, int x0, int x1)
#    @cython.staticmethod
#    cdef fromNative(THFloatTensor *tensorC, retain=*)
{% endif %}
{% endfor %}

cdef class GlobalState:
#    cdef PyTorchState *state
    cdef lua_State *L
    cdef THGenerator *generator

