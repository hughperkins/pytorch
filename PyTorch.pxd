# GENERATED FILE, do not edit by hand



cdef extern from "THRandom.h":
    cdef struct THGenerator

cdef extern from "nnWrapper.h":
    cdef struct lua_State

#cdef struct lua_State
#cdef struct THGenerator




cdef extern from "THTensor.h":
    cdef struct THDoubleTensor

cdef class _DoubleTensor(object):
    cdef THDoubleTensor *thDoubleTensor
    cpdef int dims(self)
    cpdef set1d(self, int x0, double value)
    cpdef set2d(self, int x0, int x1, double value)
    cpdef double get1d(self, int x0)
    cpdef double get2d(self, int x0, int x1)
#    @cython.staticmethod
#    cdef fromNative(THDoubleTensor *tensorC, retain=*)



cdef extern from "THTensor.h":
    cdef struct THFloatTensor

cdef class _FloatTensor(object):
    cdef THFloatTensor *thFloatTensor
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
    cdef THLongTensor *thLongTensor
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
