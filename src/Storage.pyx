# GENERATED FILE, do not edit by hand
# Source: src/Storage.jinja2.pyx

from __future__ import print_function

import cython
cimport cython

cimport cpython.array
import array

from math import log10, floor

from Storage cimport *
from nnWrapper cimport *
cimport PyTorch





cdef class DoubleStorage(object):
    # properties in .pxd file of same name

    def __init__(self, *args, **kwargs):
#        print('floatStorage.__cinit__')
        if len(args) > 0:
            raise Exception('cannot provide arguments to initializer')
        if len(kwargs) > 0:
            raise Exception('cannot provide arguments to initializer')

    @staticmethod
    def new():
#        print('allocate storage')
        return DoubleStorage_fromNative(THDoubleStorage_new(), retain=False)

    @staticmethod
    def newWithData(double [:] data):
        cdef THDoubleStorage *storageC = THDoubleStorage_newWithData(&data[0], len(data))
#        print('allocate storage')
        return DoubleStorage_fromNative(storageC, retain=False)

    @property
    def refCount(DoubleStorage self):
        return THDoubleStorage_getRefCount(self.thDoubleStorage)

    def dataAddr(DoubleStorage self):
        cdef double *data = THDoubleStorage_data(self.thDoubleStorage)
        cdef long dataAddr = pointerAsInt(data)
        return dataAddr

    @staticmethod
    def newWithSize(long size):
        cdef THDoubleStorage *storageC = THDoubleStorage_newWithSize(size)
#        print('allocate storage')
        return DoubleStorage_fromNative(storageC, retain=False)

    cpdef long size(self):
        return THDoubleStorage_size(self.thDoubleStorage)

    def __dealloc__(self):
#        print('THFloatStorage.dealloc, old refcount ', THFloatStorage_getRefCount(self.thFloatStorage))
#        print('   dealloc storage: ', hex(<long>(self.thFloatStorage)))
        THDoubleStorage_free(self.thDoubleStorage)

cdef DoubleStorage_fromNative(THDoubleStorage *storageC, retain=True):
    if retain:
        THDoubleStorage_retain(storageC)
    storage = DoubleStorage()
    storage.thDoubleStorage = storageC
    return storage


cdef class ByteStorage(object):
    # properties in .pxd file of same name

    def __init__(self, *args, **kwargs):
#        print('floatStorage.__cinit__')
        if len(args) > 0:
            raise Exception('cannot provide arguments to initializer')
        if len(kwargs) > 0:
            raise Exception('cannot provide arguments to initializer')

    @staticmethod
    def new():
#        print('allocate storage')
        return ByteStorage_fromNative(THByteStorage_new(), retain=False)

    @staticmethod
    def newWithData(unsigned char [:] data):
        cdef THByteStorage *storageC = THByteStorage_newWithData(&data[0], len(data))
#        print('allocate storage')
        return ByteStorage_fromNative(storageC, retain=False)

    @property
    def refCount(ByteStorage self):
        return THByteStorage_getRefCount(self.thByteStorage)

    def dataAddr(ByteStorage self):
        cdef unsigned char *data = THByteStorage_data(self.thByteStorage)
        cdef long dataAddr = pointerAsInt(data)
        return dataAddr

    @staticmethod
    def newWithSize(long size):
        cdef THByteStorage *storageC = THByteStorage_newWithSize(size)
#        print('allocate storage')
        return ByteStorage_fromNative(storageC, retain=False)

    cpdef long size(self):
        return THByteStorage_size(self.thByteStorage)

    def __dealloc__(self):
#        print('THFloatStorage.dealloc, old refcount ', THFloatStorage_getRefCount(self.thFloatStorage))
#        print('   dealloc storage: ', hex(<long>(self.thFloatStorage)))
        THByteStorage_free(self.thByteStorage)

cdef ByteStorage_fromNative(THByteStorage *storageC, retain=True):
    if retain:
        THByteStorage_retain(storageC)
    storage = ByteStorage()
    storage.thByteStorage = storageC
    return storage


cdef class FloatStorage(object):
    # properties in .pxd file of same name

    def __init__(self, *args, **kwargs):
#        print('floatStorage.__cinit__')
        if len(args) > 0:
            raise Exception('cannot provide arguments to initializer')
        if len(kwargs) > 0:
            raise Exception('cannot provide arguments to initializer')

    @staticmethod
    def new():
#        print('allocate storage')
        return FloatStorage_fromNative(THFloatStorage_new(), retain=False)

    @staticmethod
    def newWithData(float [:] data):
        cdef THFloatStorage *storageC = THFloatStorage_newWithData(&data[0], len(data))
#        print('allocate storage')
        return FloatStorage_fromNative(storageC, retain=False)

    @property
    def refCount(FloatStorage self):
        return THFloatStorage_getRefCount(self.thFloatStorage)

    def dataAddr(FloatStorage self):
        cdef float *data = THFloatStorage_data(self.thFloatStorage)
        cdef long dataAddr = pointerAsInt(data)
        return dataAddr

    @staticmethod
    def newWithSize(long size):
        cdef THFloatStorage *storageC = THFloatStorage_newWithSize(size)
#        print('allocate storage')
        return FloatStorage_fromNative(storageC, retain=False)

    cpdef long size(self):
        return THFloatStorage_size(self.thFloatStorage)

    def __dealloc__(self):
#        print('THFloatStorage.dealloc, old refcount ', THFloatStorage_getRefCount(self.thFloatStorage))
#        print('   dealloc storage: ', hex(<long>(self.thFloatStorage)))
        THFloatStorage_free(self.thFloatStorage)

cdef FloatStorage_fromNative(THFloatStorage *storageC, retain=True):
    if retain:
        THFloatStorage_retain(storageC)
    storage = FloatStorage()
    storage.thFloatStorage = storageC
    return storage


cdef class LongStorage(object):
    # properties in .pxd file of same name

    def __init__(self, *args, **kwargs):
#        print('floatStorage.__cinit__')
        if len(args) > 0:
            raise Exception('cannot provide arguments to initializer')
        if len(kwargs) > 0:
            raise Exception('cannot provide arguments to initializer')

    @staticmethod
    def new():
#        print('allocate storage')
        return LongStorage_fromNative(THLongStorage_new(), retain=False)

    @staticmethod
    def newWithData(long [:] data):
        cdef THLongStorage *storageC = THLongStorage_newWithData(&data[0], len(data))
#        print('allocate storage')
        return LongStorage_fromNative(storageC, retain=False)

    @property
    def refCount(LongStorage self):
        return THLongStorage_getRefCount(self.thLongStorage)

    def dataAddr(LongStorage self):
        cdef long *data = THLongStorage_data(self.thLongStorage)
        cdef long dataAddr = pointerAsInt(data)
        return dataAddr

    @staticmethod
    def newWithSize(long size):
        cdef THLongStorage *storageC = THLongStorage_newWithSize(size)
#        print('allocate storage')
        return LongStorage_fromNative(storageC, retain=False)

    cpdef long size(self):
        return THLongStorage_size(self.thLongStorage)

    def __dealloc__(self):
#        print('THFloatStorage.dealloc, old refcount ', THFloatStorage_getRefCount(self.thFloatStorage))
#        print('   dealloc storage: ', hex(<long>(self.thFloatStorage)))
        THLongStorage_free(self.thLongStorage)

cdef LongStorage_fromNative(THLongStorage *storageC, retain=True):
    if retain:
        THLongStorage_retain(storageC)
    storage = LongStorage()
    storage.thLongStorage = storageC
    return storage

