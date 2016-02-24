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
import logging


logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)




cdef floatToString(float floatValue):
    return '%.6g'% floatValue



cdef class _FloatStorage(object):
    # properties in .pxd file of same name

    def __init__(self, *args, **kwargs):
        # print('FloatStorage.__cinit__')
        logger.debug('FloatStorage.__cinit__')
        if len(args) > 0:
            for arg in args:
                if not isinstance(arg, int):
                    raise Exception('cannot provide arguments to initializer')
            if len(args) == 1:
                self.native = THFloatStorage_newWithSize(args[0])
            else:
                raise Exception('cannot provide arguments to initializer')
        if len(kwargs) > 0:
            raise Exception('cannot provide arguments to initializer')

    def __repr__(_FloatStorage self):
        cdef int size0
        size0 = THFloatStorage_size(self.native)
        res = ''
#        thisline = ''
        for c in range(size0):
            res += ' '
#            if c > 0:
#                thisline += ' '
            
            res += floatToString(self[c])
            
            res += '\n'
#        res += thisline + '\n'
        res += '[torch.FloatStorage of size ' + str(size0) + ']\n'
        return res

    @staticmethod
    def new():
        # print('allocate storage')
        return _FloatStorage_fromNative(THFloatStorage_new(), retain=False)

    @staticmethod
    def newWithData(float [:] data):
        cdef THFloatStorage *storageC = THFloatStorage_newWithData(&data[0], len(data))
        # print('allocate storage')
        return _FloatStorage_fromNative(storageC, retain=False)

    @property
    def refCount(_FloatStorage self):
        return THFloatStorage_getRefCount(self.native)

    def dataAddr(_FloatStorage self):
        cdef float *data = THFloatStorage_data(self.native)
        cdef long dataAddr = pointerAsInt(data)
        return dataAddr

    @staticmethod
    def newWithSize(long size):
        cdef THFloatStorage *storageC = THFloatStorage_newWithSize(size)
        # print('allocate storage')
        return _FloatStorage_fromNative(storageC, retain=False)

    cpdef long size(self):
        return THFloatStorage_size(self.native)

    def __len__(self):
        return self.size()

    def __dealloc__(self):
        # print('THFloatStorage.dealloc, old refcount ', THFloatStorage_getRefCount(self.thFloatStorage))
        # print('   dealloc storage: ', hex(<long>(self.thFloatStorage)))
        THFloatStorage_free(self.native)

    def __iter__(self):
        cdef int size0
        size0 = THFloatStorage_size(self.native)
        for c in range(size0):
            yield self[c]

    def __getitem__(_FloatStorage self, int index):
        cdef float res = THFloatStorage_get(self.native, index)
        return res

    def __setitem__(_FloatStorage self, int index, float value):
        THFloatStorage_set(self.native, index, value)


cdef _FloatStorage_fromNative(THFloatStorage *storageC, retain=True):
    if retain:
        THFloatStorage_retain(storageC)
    storage = _FloatStorage()
    storage.native = storageC
    return storage


cdef class _ByteStorage(object):
    # properties in .pxd file of same name

    def __init__(self, *args, **kwargs):
        # print('ByteStorage.__cinit__')
        logger.debug('ByteStorage.__cinit__')
        if len(args) > 0:
            for arg in args:
                if not isinstance(arg, int):
                    raise Exception('cannot provide arguments to initializer')
            if len(args) == 1:
                self.native = THByteStorage_newWithSize(args[0])
            else:
                raise Exception('cannot provide arguments to initializer')
        if len(kwargs) > 0:
            raise Exception('cannot provide arguments to initializer')

    def __repr__(_ByteStorage self):
        cdef int size0
        size0 = THByteStorage_size(self.native)
        res = ''
#        thisline = ''
        for c in range(size0):
            res += ' '
#            if c > 0:
#                thisline += ' '
            
            res += str(self[c])
            
            res += '\n'
#        res += thisline + '\n'
        res += '[torch.ByteStorage of size ' + str(size0) + ']\n'
        return res

    @staticmethod
    def new():
        # print('allocate storage')
        return _ByteStorage_fromNative(THByteStorage_new(), retain=False)

    @staticmethod
    def newWithData(unsigned char [:] data):
        cdef THByteStorage *storageC = THByteStorage_newWithData(&data[0], len(data))
        # print('allocate storage')
        return _ByteStorage_fromNative(storageC, retain=False)

    @property
    def refCount(_ByteStorage self):
        return THByteStorage_getRefCount(self.native)

    def dataAddr(_ByteStorage self):
        cdef unsigned char *data = THByteStorage_data(self.native)
        cdef long dataAddr = pointerAsInt(data)
        return dataAddr

    @staticmethod
    def newWithSize(long size):
        cdef THByteStorage *storageC = THByteStorage_newWithSize(size)
        # print('allocate storage')
        return _ByteStorage_fromNative(storageC, retain=False)

    cpdef long size(self):
        return THByteStorage_size(self.native)

    def __len__(self):
        return self.size()

    def __dealloc__(self):
        # print('THByteStorage.dealloc, old refcount ', THByteStorage_getRefCount(self.thByteStorage))
        # print('   dealloc storage: ', hex(<long>(self.thByteStorage)))
        THByteStorage_free(self.native)

    def __iter__(self):
        cdef int size0
        size0 = THByteStorage_size(self.native)
        for c in range(size0):
            yield self[c]

    def __getitem__(_ByteStorage self, int index):
        cdef unsigned char res = THByteStorage_get(self.native, index)
        return res

    def __setitem__(_ByteStorage self, int index, unsigned char value):
        THByteStorage_set(self.native, index, value)


cdef _ByteStorage_fromNative(THByteStorage *storageC, retain=True):
    if retain:
        THByteStorage_retain(storageC)
    storage = _ByteStorage()
    storage.native = storageC
    return storage


cdef class _LongStorage(object):
    # properties in .pxd file of same name

    def __init__(self, *args, **kwargs):
        # print('LongStorage.__cinit__')
        logger.debug('LongStorage.__cinit__')
        if len(args) > 0:
            for arg in args:
                if not isinstance(arg, int):
                    raise Exception('cannot provide arguments to initializer')
            if len(args) == 1:
                self.native = THLongStorage_newWithSize(args[0])
            else:
                raise Exception('cannot provide arguments to initializer')
        if len(kwargs) > 0:
            raise Exception('cannot provide arguments to initializer')

    def __repr__(_LongStorage self):
        cdef int size0
        size0 = THLongStorage_size(self.native)
        res = ''
#        thisline = ''
        for c in range(size0):
            res += ' '
#            if c > 0:
#                thisline += ' '
            
            res += str(self[c])
            
            res += '\n'
#        res += thisline + '\n'
        res += '[torch.LongStorage of size ' + str(size0) + ']\n'
        return res

    @staticmethod
    def new():
        # print('allocate storage')
        return _LongStorage_fromNative(THLongStorage_new(), retain=False)

    @staticmethod
    def newWithData(long [:] data):
        cdef THLongStorage *storageC = THLongStorage_newWithData(&data[0], len(data))
        # print('allocate storage')
        return _LongStorage_fromNative(storageC, retain=False)

    @property
    def refCount(_LongStorage self):
        return THLongStorage_getRefCount(self.native)

    def dataAddr(_LongStorage self):
        cdef long *data = THLongStorage_data(self.native)
        cdef long dataAddr = pointerAsInt(data)
        return dataAddr

    @staticmethod
    def newWithSize(long size):
        cdef THLongStorage *storageC = THLongStorage_newWithSize(size)
        # print('allocate storage')
        return _LongStorage_fromNative(storageC, retain=False)

    cpdef long size(self):
        return THLongStorage_size(self.native)

    def __len__(self):
        return self.size()

    def __dealloc__(self):
        # print('THLongStorage.dealloc, old refcount ', THLongStorage_getRefCount(self.thLongStorage))
        # print('   dealloc storage: ', hex(<long>(self.thLongStorage)))
        THLongStorage_free(self.native)

    def __iter__(self):
        cdef int size0
        size0 = THLongStorage_size(self.native)
        for c in range(size0):
            yield self[c]

    def __getitem__(_LongStorage self, int index):
        cdef long res = THLongStorage_get(self.native, index)
        return res

    def __setitem__(_LongStorage self, int index, long value):
        THLongStorage_set(self.native, index, value)


cdef _LongStorage_fromNative(THLongStorage *storageC, retain=True):
    if retain:
        THLongStorage_retain(storageC)
    storage = _LongStorage()
    storage.native = storageC
    return storage


cdef class _DoubleStorage(object):
    # properties in .pxd file of same name

    def __init__(self, *args, **kwargs):
        # print('DoubleStorage.__cinit__')
        logger.debug('DoubleStorage.__cinit__')
        if len(args) > 0:
            for arg in args:
                if not isinstance(arg, int):
                    raise Exception('cannot provide arguments to initializer')
            if len(args) == 1:
                self.native = THDoubleStorage_newWithSize(args[0])
            else:
                raise Exception('cannot provide arguments to initializer')
        if len(kwargs) > 0:
            raise Exception('cannot provide arguments to initializer')

    def __repr__(_DoubleStorage self):
        cdef int size0
        size0 = THDoubleStorage_size(self.native)
        res = ''
#        thisline = ''
        for c in range(size0):
            res += ' '
#            if c > 0:
#                thisline += ' '
            
            res += floatToString(self[c])
            
            res += '\n'
#        res += thisline + '\n'
        res += '[torch.DoubleStorage of size ' + str(size0) + ']\n'
        return res

    @staticmethod
    def new():
        # print('allocate storage')
        return _DoubleStorage_fromNative(THDoubleStorage_new(), retain=False)

    @staticmethod
    def newWithData(double [:] data):
        cdef THDoubleStorage *storageC = THDoubleStorage_newWithData(&data[0], len(data))
        # print('allocate storage')
        return _DoubleStorage_fromNative(storageC, retain=False)

    @property
    def refCount(_DoubleStorage self):
        return THDoubleStorage_getRefCount(self.native)

    def dataAddr(_DoubleStorage self):
        cdef double *data = THDoubleStorage_data(self.native)
        cdef long dataAddr = pointerAsInt(data)
        return dataAddr

    @staticmethod
    def newWithSize(long size):
        cdef THDoubleStorage *storageC = THDoubleStorage_newWithSize(size)
        # print('allocate storage')
        return _DoubleStorage_fromNative(storageC, retain=False)

    cpdef long size(self):
        return THDoubleStorage_size(self.native)

    def __len__(self):
        return self.size()

    def __dealloc__(self):
        # print('THDoubleStorage.dealloc, old refcount ', THDoubleStorage_getRefCount(self.thDoubleStorage))
        # print('   dealloc storage: ', hex(<long>(self.thDoubleStorage)))
        THDoubleStorage_free(self.native)

    def __iter__(self):
        cdef int size0
        size0 = THDoubleStorage_size(self.native)
        for c in range(size0):
            yield self[c]

    def __getitem__(_DoubleStorage self, int index):
        cdef double res = THDoubleStorage_get(self.native, index)
        return res

    def __setitem__(_DoubleStorage self, int index, double value):
        THDoubleStorage_set(self.native, index, value)


cdef _DoubleStorage_fromNative(THDoubleStorage *storageC, retain=True):
    if retain:
        THDoubleStorage_retain(storageC)
    storage = _DoubleStorage()
    storage.native = storageC
    return storage

