import cython
cimport cython

#import numpy
#cimport numpy

#from cpython import array
cimport cpython.array
import array

# class Tensor(object):
#     def __init__(self, numpy_array):
#         print('Tensor.__init__')

cdef extern from "mylib.h":
    cdef float mysum(int rows, int cols, float *myarray)

def process1(int rows, int cols, float[:] myarray):
    print('process1')
#    dims = len(array.shape)
#    print('dims', dims)
#    rows = array.shape[0]
#    cols = array.shape[1]
#    print('rows=' + str(rows) + ' cols=' + str(cols))

#    A = array.array('f', [3] * 2 * 3)
    res = mysum(rows, cols, &myarray[0])
    return res

def process2(myarray):
    print('process2')
    dims = len(myarray.shape)
    print('dims', dims)
    rows = myarray.shape[0]
    cols = myarray.shape[1]
    print('rows=' + str(rows) + ' cols=' + str(cols))

#    A = array.array('f', [3] * 2 * 3)
    cdef float[:] myarraymv = myarray.reshape(rows * cols)
    res = mysum(rows, cols, &myarraymv[0])
    return res

#cdef struct THFloatStorage:
#    pass

cdef extern from "THStorage.h":
    cdef struct THFloatStorage
    THFloatStorage* THFloatStorage_newWithData(float *data, long size)
    THFloatStorage* THFloatStorage_new()

#cdef struct THFloatTensor:
#    pass

cdef extern from "THTensor.h":
    cdef struct THFloatTensor
    THFloatTensor* THFloatTensor_newWithStorage2d(THFloatStorage *storage, long storageOffset, long size0, long stride0, long size1, long stride1)
    void THFloatTensor_add(THFloatTensor *tensorSelf, THFloatTensor *tensorOne, float value)
    void THFloatTensor_addmm(THFloatTensor *tensorSelf, float beta, THFloatTensor *tensorOne, float alpha, THFloatTensor *mat1, THFloatTensor *mat2)
    int THFloatTensor_nDimension(THFloatTensor *tensor)
    long THFloatTensor_size(const THFloatTensor *self, int dim)
    long THFloatTensor_stride(const THFloatTensor *self, int dim)
    float THFloatTensor_get2d(const THFloatTensor *tensor, long x0, long x1)
    void THFloatTensor_set2d(const THFloatTensor *tensor, long x0, long x1, float value)
    void THFloatTensor_add(THFloatTensor *r_, THFloatTensor *t, float value)

def process3(myarray):
    print('process2')
    dims = len(myarray.shape)
    print('dims', dims)
    rows = myarray.shape[0]
    cols = myarray.shape[1]
    print('rows=' + str(rows) + ' cols=' + str(cols))

#    A = array.array('f', [3] * 2 * 3)
    cdef float[:] myarraymv = myarray.reshape(rows * cols)
    cdef THFloatStorage *floatStorage = THFloatStorage_newWithData(&myarraymv[0], rows * cols)
    cdef THFloatTensor *tensor = THFloatTensor_newWithStorage2d(floatStorage, 0, rows, cols, cols, 1)
    THFloatTensor_add(tensor, tensor, 19)

cdef class Storage(object):
    cdef THFloatStorage *thFloatStorage

    def __cinit__(self, myarray=None):
        cdef float[:] myarraymv
        if myarray is not None:
            dims = len(myarray.shape)
            print('dims', dims)
            rows = myarray.shape[0]
            cols = myarray.shape[1]
            print('rows=' + str(rows) + ' cols=' + str(cols))

    #    A = array.array('f', [3] * 2 * 3)
            myarraymv = myarray.reshape(rows * cols)
            self.thFloatStorage = THFloatStorage_newWithData(&myarraymv[0], rows * cols)
        else:
            self.thFloatStorage = THFloatStorage_new()

    def __dealloc__(self):
        pass

cdef class Tensor(object):
    cdef THFloatTensor *thFloatTensor
    cdef Storage storage

    def __cinit__(self, Storage storage, offset, size0, stride0, size1, stride1):
        self.thFloatTensor = THFloatTensor_newWithStorage2d(storage.thFloatStorage, offset, size0, stride0, size1, stride1)
        self.storage = storage

    def __dealloc__(self):
        pass

    cpdef int dims(self):
        return THFloatTensor_nDimension(self.thFloatTensor)

    cpdef set2d(self, int x0, int x1, float value):
        THFloatTensor_set2d(self.thFloatTensor, x0, x1, value)

    cpdef float get2d(self, int x0, int x1):
        return THFloatTensor_get2d(self.thFloatTensor, x0, x1)

    def __add__(Tensor self, float value):
        THFloatTensor_add(self.thFloatTensor, self.thFloatTensor, value)

#    def __mul__(Tensor self, Tensor M2):
##        Tensor T = Tensor()
#        cdef array Tarray = array('f', [0] * 
#        THFloatTensor_addmm(self.thFloatTensor, 0, NULL, 1, self.thFloatTensor, M2.thFloatTensor)
#        return self

def asTensor(myarray):
    print('process2')
    dims = len(myarray.shape)
    print('dims', dims)
    rows = myarray.shape[0]
    cols = myarray.shape[1]
    print('rows=' + str(rows) + ' cols=' + str(cols))

#    A = array.array('f', [3] * 2 * 3)
    cdef float[:] myarraymv = myarray.reshape(rows * cols)
    storage = Storage(myarray)
    tensor = Tensor(storage, 0, rows, cols, cols, 1)
    return tensor

