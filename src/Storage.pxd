


cdef extern from "nnWrapper.h":
    int THDoubleStorage_getRefCount(THDoubleStorage *self)

cdef extern from "nnWrapper.h":
    int THByteStorage_getRefCount(THByteStorage *self)

cdef extern from "nnWrapper.h":
    int THFloatStorage_getRefCount(THFloatStorage *self)

cdef extern from "nnWrapper.h":
    int THLongStorage_getRefCount(THLongStorage *self)




cdef extern from "THStorage.h":
    cdef struct THDoubleStorage
    THDoubleStorage* THDoubleStorage_newWithData(double *data, long size)
    THDoubleStorage* THDoubleStorage_new()
    THDoubleStorage* THDoubleStorage_newWithSize(long size)
    double *THDoubleStorage_data(THDoubleStorage *self)
    long THDoubleStorage_size(THDoubleStorage *self)
    void THDoubleStorage_free(THDoubleStorage *self)
    void THDoubleStorage_retain(THDoubleStorage *self)


cdef extern from "THStorage.h":
    cdef struct THByteStorage
    THByteStorage* THByteStorage_newWithData(unsigned char *data, long size)
    THByteStorage* THByteStorage_new()
    THByteStorage* THByteStorage_newWithSize(long size)
    unsigned char *THByteStorage_data(THByteStorage *self)
    long THByteStorage_size(THByteStorage *self)
    void THByteStorage_free(THByteStorage *self)
    void THByteStorage_retain(THByteStorage *self)


cdef extern from "THStorage.h":
    cdef struct THFloatStorage
    THFloatStorage* THFloatStorage_newWithData(float *data, long size)
    THFloatStorage* THFloatStorage_new()
    THFloatStorage* THFloatStorage_newWithSize(long size)
    float *THFloatStorage_data(THFloatStorage *self)
    long THFloatStorage_size(THFloatStorage *self)
    void THFloatStorage_free(THFloatStorage *self)
    void THFloatStorage_retain(THFloatStorage *self)


cdef extern from "THStorage.h":
    cdef struct THLongStorage
    THLongStorage* THLongStorage_newWithData(long *data, long size)
    THLongStorage* THLongStorage_new()
    THLongStorage* THLongStorage_newWithSize(long size)
    long *THLongStorage_data(THLongStorage *self)
    long THLongStorage_size(THLongStorage *self)
    void THLongStorage_free(THLongStorage *self)
    void THLongStorage_retain(THLongStorage *self)




cdef class DoubleStorage(object):
    cdef THDoubleStorage *thDoubleStorage
    cpdef long size(self)

cdef DoubleStorage_fromNative(THDoubleStorage *storageC, retain=*)


cdef class ByteStorage(object):
    cdef THByteStorage *thByteStorage
    cpdef long size(self)

cdef ByteStorage_fromNative(THByteStorage *storageC, retain=*)


cdef class FloatStorage(object):
    cdef THFloatStorage *thFloatStorage
    cpdef long size(self)

cdef FloatStorage_fromNative(THFloatStorage *storageC, retain=*)


cdef class LongStorage(object):
    cdef THLongStorage *thLongStorage
    cpdef long size(self)

cdef LongStorage_fromNative(THLongStorage *storageC, retain=*)

