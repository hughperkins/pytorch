# GENERATED FILE, do not edit by hand
# Source: src/Storage.jinja2.pxd




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
    void THDoubleStorage_set(THDoubleStorage*, long, double)
    double THDoubleStorage_get(const THDoubleStorage*, long)



cdef extern from "THStorage.h":
    cdef struct THByteStorage
    THByteStorage* THByteStorage_newWithData(unsigned char *data, long size)
    THByteStorage* THByteStorage_new()
    THByteStorage* THByteStorage_newWithSize(long size)
    unsigned char *THByteStorage_data(THByteStorage *self)
    long THByteStorage_size(THByteStorage *self)
    void THByteStorage_free(THByteStorage *self)
    void THByteStorage_retain(THByteStorage *self)
    void THByteStorage_set(THByteStorage*, long, unsigned char)
    unsigned char THByteStorage_get(const THByteStorage*, long)



cdef extern from "THStorage.h":
    cdef struct THFloatStorage
    THFloatStorage* THFloatStorage_newWithData(float *data, long size)
    THFloatStorage* THFloatStorage_new()
    THFloatStorage* THFloatStorage_newWithSize(long size)
    float *THFloatStorage_data(THFloatStorage *self)
    long THFloatStorage_size(THFloatStorage *self)
    void THFloatStorage_free(THFloatStorage *self)
    void THFloatStorage_retain(THFloatStorage *self)
    void THFloatStorage_set(THFloatStorage*, long, float)
    float THFloatStorage_get(const THFloatStorage*, long)



cdef extern from "THStorage.h":
    cdef struct THLongStorage
    THLongStorage* THLongStorage_newWithData(long *data, long size)
    THLongStorage* THLongStorage_new()
    THLongStorage* THLongStorage_newWithSize(long size)
    long *THLongStorage_data(THLongStorage *self)
    long THLongStorage_size(THLongStorage *self)
    void THLongStorage_free(THLongStorage *self)
    void THLongStorage_retain(THLongStorage *self)
    void THLongStorage_set(THLongStorage*, long, long)
    long THLongStorage_get(const THLongStorage*, long)





cdef class _DoubleStorage(object):
    cdef THDoubleStorage *native
    cpdef long size(self)

cdef _DoubleStorage_fromNative(THDoubleStorage *storageC, retain=*)


cdef class _ByteStorage(object):
    cdef THByteStorage *native
    cpdef long size(self)

cdef _ByteStorage_fromNative(THByteStorage *storageC, retain=*)


cdef class _FloatStorage(object):
    cdef THFloatStorage *native
    cpdef long size(self)

cdef _FloatStorage_fromNative(THFloatStorage *storageC, retain=*)


cdef class _LongStorage(object):
    cdef THLongStorage *native
    cpdef long size(self)

cdef _LongStorage_fromNative(THLongStorage *storageC, retain=*)

