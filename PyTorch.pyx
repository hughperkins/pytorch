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

#cdef struct THFloatTensor:
#    pass

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

cdef extern from "THStorage.h":
    cdef struct THFloatStorage
    THFloatStorage* THFloatStorage_newWithData(float *data, long size)
    THFloatStorage* THFloatStorage_new()
    THFloatStorage* THFloatStorage_newWithSize(long size)
    long THFloatStorage_size(THFloatStorage *self)
    void THFloatStorage_free(THFloatStorage *self)

cdef class Storage(object):
    cdef THFloatStorage *thFloatStorage

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            raise Exception('cannot provide arguments to initializer')
        if len(kwargs) > 0:
            raise Exception('cannot provide arguments to initializer')

    @staticmethod
    def new():
        cdef THFloatStorage *storageC = THFloatStorage_new()
        print('allocate storage')
        storage = Storage()
        storage.thFloatStorage = storageC
        return storage

    @staticmethod
    def newWithData(float [:] data):
        cdef THFloatStorage *storageC = THFloatStorage_newWithData(&data[0], len(data))
        print('allocate storage')
        storage = Storage()
        storage.thFloatStorage = storageC
        return storage

    @staticmethod
    def newWithSize(long size):
        cdef THFloatStorage *storageC = THFloatStorage_newWithSize(size)
        print('allocate storage')
        storage = Storage()
        storage.thFloatStorage = storageC
        return storage

    cpdef long size(self):
        return THFloatStorage_size(self.thFloatStorage)

    def __dealloc__(self):
        print('free storage')
        THFloatStorage_free(self.thFloatStorage)

cdef extern from "THTensor.h":
    cdef struct THFloatTensor
    THFloatTensor* THFloatTensor_newWithStorage2d(THFloatStorage *storage, long storageOffset, long size0, long stride0, long size1, long stride1)
    void THFloatTensor_add(THFloatTensor *tensorSelf, THFloatTensor *tensorOne, float value)
    void THFloatTensor_addmm(THFloatTensor *tensorSelf, float beta, THFloatTensor *tensorOne, float alpha, THFloatTensor *mat1, THFloatTensor *mat2)
    int THFloatTensor_nDimension(THFloatTensor *tensor)
    THFloatTensor *THFloatTensor_new()
    THFloatTensor *THFloatTensor_newWithSize2d(long size0, long size1)
    void THFloatTensor_free(THFloatTensor *self)
    void THFloatTensor_resizeAs(THFloatTensor *self, THFloatTensor *model)
    void THFloatTensor_resize2d(THFloatTensor *self, long size0, long size1)
    long THFloatTensor_size(const THFloatTensor *self, int dim)
    long THFloatTensor_stride(const THFloatTensor *self, int dim)
    float THFloatTensor_get2d(const THFloatTensor *tensor, long x0, long x1)
    void THFloatTensor_set2d(const THFloatTensor *tensor, long x0, long x1, float value)
    void THFloatTensor_add(THFloatTensor *r_, THFloatTensor *t, float value)
    THFloatStorage *THFloatTensor_storage(THFloatTensor *self)

cdef class Tensor(object):
    cdef THFloatTensor *thFloatTensor
    cdef Storage storage

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            raise Exception('cannot provide arguments to initializer')
        if len(kwargs) > 0:
            raise Exception('cannot provide arguments to initializer')

#    def __cinit__(self, THFloatTensor *tensorC, Storage storage):
#        self.thFloatTensor = tensorC
#        self.storage = storage

#    def __cinit__(self, Storage storage, offset, size0, stride0, size1, stride1):
#        self.thFloatTensor = THFloatTensor_newWithStorage2d(storage.thFloatStorage, offset, size0, stride0, size1, stride1)
#        self.storage = storage

    def __dealloc__(self):
        print('free tensor')
        THFloatTensor_free(self.thFloatTensor)

    cpdef int dims(self):
        return THFloatTensor_nDimension(self.thFloatTensor)

    cpdef set2d(self, int x0, int x1, float value):
        THFloatTensor_set2d(self.thFloatTensor, x0, x1, value)

    cpdef float get2d(self, int x0, int x1):
        return THFloatTensor_get2d(self.thFloatTensor, x0, x1)

    @staticmethod
    def new():
        print('allocate tensor')
        cdef THFloatTensor *newTensorC = THFloatTensor_new()
        tensor = Tensor()
        tensor.thFloatTensor = newTensorC
        cdef THFloatStorage *storageC = THFloatTensor_storage(newTensorC)
        if storageC != NULL:
            tensor.storage = Storage()
            tensor.storage.thFloatStorage = storageC
        return tensor

    @staticmethod
    def newWithStorage2d(Storage storage, offset, size0, stride0, size1, stride1):
        print('allocate tensor')
        cdef THFloatTensor *newTensorC = THFloatTensor_newWithStorage2d(storage.thFloatStorage, offset, size0, stride0, size1, stride1)
        tensor = Tensor()
        tensor.thFloatTensor = newTensorC
        tensor.storage = storage
        return tensor        

    def resize2d(Tensor self, long size0, long size1):
        newNumElements = size0 * size1
        currentNumElements = 0
        if self.storage is not None:
            currentNumElements = self.storage.size()
        if currentNumElements > newNumElements:
            self.thFloatTensor = THFloatTensor_newWithSize2d(size0, size1)
            self.storage = Storage()
            self.storage.thFloatStorage = THFloatTensor_storage(self.thFloatTensor)
        else:
            THFloatTensor_resize2d(self.thFloatTensor, size0, size1)

        return self
#            this.storage = Storage.newWithSize(newNumElements)
        
    def __iadd__(Tensor self, float value):
        print('iadd')
        THFloatTensor_add(self.thFloatTensor, self.thFloatTensor, value)
        return self

    def __add__(Tensor self, float value):
        print('iadd')
        # assume 2d matrix for now?
        cdef Tensor res = Tensor.new()
#        THFloatTensor_resizeAs(cresult, self.thFloatTensor)
        THFloatTensor_add(res.thFloatTensor, self.thFloatTensor, value)
        return res

    def __mul__(Tensor self, Tensor M2):
        cdef Tensor T = Tensor.new()
        cdef Tensor res = Tensor.new()
        cdef int resRows = THFloatTensor_size(self.thFloatTensor, 0)
        cdef int resCols = THFloatTensor_size(M2.thFloatTensor, 1)
        res.resize2d(resRows, resCols)
        T.resize2d(resRows, resCols)
#        cdef array Tarray = array('f', [0] * 
        THFloatTensor_addmm(res.thFloatTensor, 0, T.thFloatTensor, 1, self.thFloatTensor, M2.thFloatTensor)
        return res

    def __repr__(Tensor self):
        # assume 2d matrix for now
#        print('tensorAB get2d:')
        cdef int rows = THFloatTensor_size(self.thFloatTensor, 0)
        cdef int cols = THFloatTensor_size(self.thFloatTensor, 1)
        res = ''
        for r in range(rows):
            thisline = ''
            for c in range(cols):
                if c > 0:
                    thisline += ' '
                thisline += str(self.get2d(r,c))
            res += thisline + '\n'
        res += '[torch.FloatTensor of size ' + str(rows) + 'x' + str(cols) + ']\n'
        return res

def asTensor(myarray):
    print('process2')
    dims = len(myarray.shape)
    print('dims', dims)
    rows = myarray.shape[0]
    cols = myarray.shape[1]
    print('rows=' + str(rows) + ' cols=' + str(cols))

#    A = array.array('f', [3] * 2 * 3)
    cdef float[:] myarraymv = myarray.reshape(rows * cols)
    storage = Storage.newWithData(myarraymv)
#    tensor = Tensor(storage, 0, rows, cols, cols, 1)
    tensor = Tensor.newWithStorage2d(storage, 0, rows, cols, cols, 1)
    return tensor

