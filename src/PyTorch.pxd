# GENERATED FILE, do not edit by hand
# Source: src/PyTorch.jinja2.pxd



cdef extern from "THRandom.h":
    cdef struct THGenerator

cdef extern from "nnWrapper.h":
    cdef struct lua_State

#cdef struct lua_State
#cdef struct THGenerator




cdef extern from "THTensor.h":
    cdef struct THDoubleTensor

cdef class _DoubleTensor(object):
    cdef THDoubleTensor *native
    cpdef int dims(self)
    cpdef set1d(self, int x0, double value)
    cpdef set2d(self, int x0, int x1, double value)
    cpdef double get1d(self, int x0)
    cpdef double get2d(self, int x0, int x1)
#    @cython.staticmethod
#    cdef fromNative(THDoubleTensor *tensorC, retain=*)



cdef extern from "THTensor.h":
    cdef struct THByteTensor

cdef class _ByteTensor(object):
    cdef THByteTensor *native
    cpdef int dims(self)
    cpdef set1d(self, int x0, unsigned char value)
    cpdef set2d(self, int x0, int x1, unsigned char value)
    cpdef unsigned char get1d(self, int x0)
    cpdef unsigned char get2d(self, int x0, int x1)
#    @cython.staticmethod
#    cdef fromNative(THByteTensor *tensorC, retain=*)



cdef extern from "THTensor.h":
    cdef struct THFloatTensor

cdef class _FloatTensor(object):
    cdef THFloatTensor *native
    cpdef int dims(self)
    cpdef set1d(self, int x0, float value)
    cpdef set2d(self, int x0, int x1, float value)
    cpdef float get1d(self, int x0)
    cpdef float get2d(self, int x0, int x1)
#    @cython.staticmethod
#    cdef fromNative(THFloatTensor *tensorC, retain=*)



cdef extern from "THTensor.h":
    cdef struct THLongTensor

cdef class _LongTensor(object):
    cdef THLongTensor *native
    cpdef int dims(self)
    cpdef set1d(self, int x0, long value)
    cpdef set2d(self, int x0, int x1, long value)
    cpdef long get1d(self, int x0)
    cpdef long get2d(self, int x0, int x1)
#    @cython.staticmethod
#    cdef fromNative(THLongTensor *tensorC, retain=*)


cdef class GlobalState:
#    cdef PyTorchState *state
    cdef lua_State *L
    cdef THGenerator *generator
