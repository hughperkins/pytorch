import cython
cimport cython

cimport cpython.array
import array

cdef extern from "THStorage.h":
    cdef struct THFloatStorage
    THFloatStorage* THFloatStorage_newWithData(float *data, long size)
    THFloatStorage* THFloatStorage_new()
    THFloatStorage* THFloatStorage_newWithSize(long size)
    long THFloatStorage_size(THFloatStorage *self)
    void THFloatStorage_free(THFloatStorage *self)
    void THFloatStorage_retain(THFloatStorage *self)

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
    THFloatTensor *THFloatTensor_newWithSize1d(long size0)
    THFloatTensor *THFloatTensor_newWithSize2d(long size0, long size1)
    THFloatTensor *THFloatTensor_newSelect(THFloatTensor *self, int dimension, int sliceIndex)
    void THFloatTensor_free(THFloatTensor *self)
    void THFloatTensor_resizeAs(THFloatTensor *self, THFloatTensor *model)
    void THFloatTensor_resize2d(THFloatTensor *self, long size0, long size1)
    long THFloatTensor_size(const THFloatTensor *self, int dim)
    long THFloatTensor_stride(const THFloatTensor *self, int dim)
    float THFloatTensor_get1d(const THFloatTensor *tensor, long x0)
    float THFloatTensor_get2d(const THFloatTensor *tensor, long x0, long x1)
    void THFloatTensor_set1d(const THFloatTensor *tensor, long x0, float value)
    void THFloatTensor_set2d(const THFloatTensor *tensor, long x0, long x1, float value)
    void THFloatTensor_fill(THFloatTensor *self, float value)
#    void THFloatTensor_uniform(THFloatTensor *self, float value)
    void THFloatTensor_add(THFloatTensor *r_, THFloatTensor *t, float value)
    THFloatStorage *THFloatTensor_storage(THFloatTensor *self)
    void THFloatTensor_retain(THFloatTensor *self)

cdef class FloatTensor(object):
    cdef THFloatTensor *thFloatTensor

#    def __cinit__(Tensor self, THFloatTensor *tensorC = NULL):
#        self.thFloatTensor = tensorC

    def __init__(self, *args, **kwargs):
        if len(kwargs) > 0:
            raise Exception('cannot provide arguments to initializer')
        if len(args) > 0:
            for arg in args:
                if not isinstance(arg, int):
                    raise Exception('cannot provide arguments to initializer')
            if len(args) == 1:
                self.thFloatTensor = THFloatTensor_newWithSize1d(args[0])
            if len(args) == 2:
                self.thFloatTensor = THFloatTensor_newWithSize2d(args[0], args[1])
            else:
                raise Exception('Not implemented')

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

    cpdef set1d(self, int x0, float value):
        THFloatTensor_set1d(self.thFloatTensor, x0, value)

    cpdef set2d(self, int x0, int x1, float value):
        THFloatTensor_set2d(self.thFloatTensor, x0, x1, value)

    cpdef float get1d(self, int x0):
        return THFloatTensor_get1d(self.thFloatTensor, x0)

    cpdef float get2d(self, int x0, int x1):
        return THFloatTensor_get2d(self.thFloatTensor, x0, x1)

    @staticmethod
    cdef fromNative(THFloatTensor *tensorC):
        tensor = FloatTensor()
        tensor.thFloatTensor = tensorC
        return tensor

    @staticmethod
    def new():
        print('allocate tensor')
        cdef THFloatTensor *newTensorC = THFloatTensor_new()
        return FloatTensor.fromNative(newTensorC)

    @staticmethod
    def newWithStorage2d(Storage storage, offset, size0, stride0, size1, stride1):
        print('allocate tensor')
        cdef THFloatTensor *newTensorC = THFloatTensor_newWithStorage2d(storage.thFloatStorage, offset, size0, stride0, size1, stride1)
        return FloatTensor.fromNative(newTensorC)

    def resize2d(FloatTensor self, long size0, long size1):
        THFloatTensor_resize2d(self.thFloatTensor, size0, size1)
        return self
        
    def __iadd__(FloatTensor self, float value):
        print('iadd')
        THFloatTensor_add(self.thFloatTensor, self.thFloatTensor, value)
        return self

    def __getitem__(FloatTensor self, int index):
        cdef THFloatTensor *res = THFloatTensor_newSelect(self.thFloatTensor, 0, index)
        return FloatTensor.fromNative(res)

    def __add__(FloatTensor self, float value):
        print('iadd')
        # assume 2d matrix for now?
        cdef FloatTensor res = FloatTensor.new()
#        THFloatTensor_resizeAs(cresult, self.thFloatTensor)
        THFloatTensor_add(res.thFloatTensor, self.thFloatTensor, value)
        return res

    def fill(FloatTensor self, float value):
        THFloatTensor_fill(self.thFloatTensor, value)
        return self

    def size(FloatTensor self):
        cdef int dims = self.dims()
        cdef FloatTensor size = FloatTensor(dims)
        for d in range(dims):
            size.set1d(d, THFloatTensor_size(self.thFloatTensor, d))
        return size

    def __mul__(FloatTensor self, FloatTensor M2):
        cdef FloatTensor T = FloatTensor.new()
        cdef FloatTensor res = FloatTensor.new()
        cdef int resRows = THFloatTensor_size(self.thFloatTensor, 0)
        cdef int resCols = THFloatTensor_size(M2.thFloatTensor, 1)
        res.resize2d(resRows, resCols)
        T.resize2d(resRows, resCols)
        THFloatTensor_addmm(res.thFloatTensor, 0, T.thFloatTensor, 1, self.thFloatTensor, M2.thFloatTensor)
        return res

    def __repr__(FloatTensor self):
        # assume 2d matrix for now
        cdef int size0
        cdef int size1
        dims = self.dims()
        if dims == 2:
            size0 = THFloatTensor_size(self.thFloatTensor, 0)
            size1 = THFloatTensor_size(self.thFloatTensor, 1)
            res = ''
            for r in range(size0):
                thisline = ''
                for c in range(size1):
                    if c > 0:
                        thisline += ' '
                    thisline += str(self.get2d(r,c))
                res += thisline + '\n'
            res += '[torch.FloatTensor of size ' + str(size0) + 'x' + str(size1) + ']\n'
            return res
        elif dims == 1:
            size0 = THFloatTensor_size(self.thFloatTensor, 0)
            res = ''
            thisline = ''
            for c in range(size0):
                if c > 0:
                    thisline += ' '
                thisline += str(self.get1d(c))
            res += thisline + '\n'
            res += '[torch.FloatTensor of size ' + str(size0) + ']\n'
            return res
        else:
            raise Exception("Not implemented: dims > 2")

def asTensor(myarray):
    dims = len(myarray.shape)
#    print('dims', dims)
    rows = myarray.shape[0]
    cols = myarray.shape[1]
#    print('rows=' + str(rows) + ' cols=' + str(cols))

    cdef float[:] myarraymv = myarray.reshape(rows * cols)
    storage = Storage.newWithData(myarraymv)
    tensor = FloatTensor.newWithStorage2d(storage, 0, rows, cols, cols, 1)
    return tensor

cdef extern from "nnWrapper.h":
    cdef struct lua_State
    lua_State *luaInit()
    void luaClose(lua_State *L)

    cdef cppclass _Module:
        THFloatTensor *updateOutput(THFloatTensor *input)
        THFloatTensor *updateGradInput(THFloatTensor *input, THFloatTensor *gradOutput)
        THFloatTensor *getOutput()

    cdef cppclass _Linear(_Module):
        _Linear(lua_State *L, int inputSize, int OutputSize)
        THFloatTensor *getWeight()

    cdef cppclass _Criterion:
        THFloatTensor *updateOutput(THFloatTensor *input)
        THFloatTensor *updateGradInput(THFloatTensor *input, THFloatTensor *target)

    cdef cppclass _MSECriterion(_Criterion):
        _MSECriterion(lua_State *L)

    cdef cppclass _Trainer:
        pass

    cdef cppclass _StochasticGradient:
        _StochasticGradient(lua_State *L, _Module *module, _Criterion *criterion)

cdef class Module(object):
    cdef _Module *native

cdef class Linear(Module):

    def __cinit__(self, Nn nn, inputSize, outputSize):
        self.native = new _Linear(nn.L, inputSize, outputSize)

    def __dealloc__(self):
        del self.native

    def updateOutput(self, FloatTensor input):
        cdef THFloatTensor *outputC = self.native.updateOutput(input.thFloatTensor)
        return FloatTensor.fromNative(outputC)

    def getOutput(self):
        return FloatTensor.fromNative(self.native.getOutput())

    def getWeight(self):
        return FloatTensor.fromNative((<_Linear *>(self.native)).getWeight())

cdef class Criterion(object):
    cdef _Criterion *native

cdef class MSECriterion(Criterion):

    def __cinit__(self, Nn nn):
        self.native = new _MSECriterion(nn.L)

    def __dealloc__(self):
        del self.native

    def updateOutput(self, FloatTensor input):
        cdef THFloatTensor *outputC = self.native.updateOutput(input.thFloatTensor)
        return FloatTensor.fromNative(outputC)

    def updateGradInput(self, FloatTensor input, FloatTensor target):
        cdef THFloatTensor *gradInputC = self.native.updateGradInput(input.thFloatTensor, target.thFloatTensor)
        return FloatTensor.fromNative(gradInputC)

cdef class StochasticGradient(object):
    cdef _StochasticGradient *native

    def __cinit__(self, Nn nn, Module module, Criterion criterion):
        self.native = new _StochasticGradient(nn.L, module.native, criterion.native)

    def __dealloc__(self):
        del self.native

cdef class Nn(object):  # basically holds the Lua state
    cdef lua_State *L

    def __cinit__(self):
        self.L = luaInit()
    
    def __dealloc__(self):
        luaClose(self.L)

    def Linear(self, inputSize, outputSize):
        cdef Linear linear = Linear(self, inputSize, outputSize)
        return linear

    def MSECriterion(self):
        return MSECriterion(self)

    def StochasticGradient(self, module, criterion):
        return StochasticGradient(self, module, criterion)

