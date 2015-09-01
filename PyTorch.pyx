import cython
cimport cython

cimport cpython.array
import array

#cdef extern from "nnWrapper.h":
#    long pointerAsInt(void *ptr)

from math import log10, floor

# from http://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)

cdef extern from "THStorage.h":
    cdef struct THFloatStorage
    THFloatStorage* THFloatStorage_newWithData(float *data, long size)
    THFloatStorage* THFloatStorage_new()
    THFloatStorage* THFloatStorage_newWithSize(long size)
    float *THFloatStorage_data(THFloatStorage *self)
    long THFloatStorage_size(THFloatStorage *self)
    void THFloatStorage_free(THFloatStorage *self)
    void THFloatStorage_retain(THFloatStorage *self)

cdef floatToString(float floatValue):
    return '%.6g'% floatValue

cdef class FloatStorage(object):
    cdef THFloatStorage *thFloatStorage

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            raise Exception('cannot provide arguments to initializer')
        if len(kwargs) > 0:
            raise Exception('cannot provide arguments to initializer')

    @staticmethod
    cdef fromNative(THFloatStorage *storageC, retain=True):
        storage = FloatStorage()
        storage.thFloatStorage = storageC
        return storage

    @staticmethod
    def new():
#        print('allocate storage')
        return FloatStorage.fromNative(THFloatStorage_new(), retain=False)

    @staticmethod
    def newWithData(float [:] data):
        cdef THFloatStorage *storageC = THFloatStorage_newWithData(&data[0], len(data))
#        print('allocate storage')
        return FloatStorage.fromNative(storageC, retain=False)

    def dataAddr(FloatStorage self):
        cdef float *data = THFloatStorage_data(self.thFloatStorage)
        cdef long dataAddr = pointerAsInt(data)
        return dataAddr

    @staticmethod
    def newWithSize(long size):
        cdef THFloatStorage *storageC = THFloatStorage_newWithSize(size)
#        print('allocate storage')
        return FloatStorage.fromNative(storageC, retain=False)

    cpdef long size(self):
        return THFloatStorage_size(self.thFloatStorage)

    def __dealloc__(self):
#        print('free storage')
        THFloatStorage_free(self.thFloatStorage)

cdef extern from "THTensor.h":
    cdef struct THFloatTensor
    THFloatTensor* THFloatTensor_newWithStorage1d(THFloatStorage *storage, long storageOffset, long size0, long stride0)
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
            elif len(args) == 2:
                self.thFloatTensor = THFloatTensor_newWithSize2d(args[0], args[1])
            else:
                raise Exception('Not implemented, len(args)=' + str(len(args)))

#    def __cinit__(self, THFloatTensor *tensorC, Storage storage):
#        self.thFloatTensor = tensorC
#        self.storage = storage

#    def __cinit__(self, Storage storage, offset, size0, stride0, size1, stride1):
#        self.thFloatTensor = THFloatTensor_newWithStorage2d(storage.thFloatStorage, offset, size0, stride0, size1, stride1)
#        self.storage = storage

    def __dealloc__(self):
#        print('free tensor')
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
    cdef fromNative(THFloatTensor *tensorC, retain=True):
        if retain:
            THFloatTensor_retain(tensorC)
        tensor = FloatTensor()
        tensor.thFloatTensor = tensorC
        return tensor

    @staticmethod
    def new():
#        print('allocate tensor')
        cdef THFloatTensor *newTensorC = THFloatTensor_new()
        return FloatTensor.fromNative(newTensorC, False)

    @staticmethod
    def newWithStorage1d(FloatStorage storage, offset, size0, stride0):
#        print('allocate tensor')
        cdef THFloatTensor *newTensorC = THFloatTensor_newWithStorage1d(storage.thFloatStorage, offset, size0, stride0)
        return FloatTensor.fromNative(newTensorC, False)

    @staticmethod
    def newWithStorage2d(FloatStorage storage, offset, size0, stride0, size1, stride1):
#        print('allocate tensor')
        cdef THFloatTensor *newTensorC = THFloatTensor_newWithStorage2d(storage.thFloatStorage, offset, size0, stride0, size1, stride1)
        return FloatTensor.fromNative(newTensorC, False)

    def resize2d(FloatTensor self, long size0, long size1):
        THFloatTensor_resize2d(self.thFloatTensor, size0, size1)
        return self

    def storage(FloatTensor self):
        cdef THFloatStorage *storageC = THFloatTensor_storage(self.thFloatTensor)
        if storageC == NULL:
            return None
        return FloatStorage.fromNative(storageC)

    def __getitem__(FloatTensor self, int index):
        if self.dims() == 1:
            return self.get1d(index)
        cdef THFloatTensor *res = THFloatTensor_newSelect(self.thFloatTensor, 0, index)
        return FloatTensor.fromNative(res, False)

    def __setitem__(FloatTensor self, int index, float value):
        if self.dims() == 1:
            self.set1d(index, value)
        else:
            raise Exception("not implemented")
#        return self
#        cdef THFloatTensor *res = THFloatTensor_newSelect(self.thFloatTensor, 0, index)
#        return FloatTensor.fromNative(res, False)

    def __iadd__(FloatTensor self, float value):
        print('iadd')
        THFloatTensor_add(self.thFloatTensor, self.thFloatTensor, value)
        return self

    def __add__(FloatTensor self, float value):
        print('add')
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
        if dims == 0:
            return '[torch.FloatTensor with no dimension]\n'
        elif dims == 2:
            size0 = THFloatTensor_size(self.thFloatTensor, 0)
            size1 = THFloatTensor_size(self.thFloatTensor, 1)
            res = ''
            for r in range(size0):
                thisline = ''
                for c in range(size1):
                    if c > 0:
                        thisline += ' '
                    thisline += floatToString(self.get2d(r,c),)
                res += thisline + '\n'
            res += '[torch.FloatTensor of size ' + ('%.0f' % size0) + 'x' + str(size1) + ']\n'
            return res
        elif dims == 1:
            size0 = THFloatTensor_size(self.thFloatTensor, 0)
            res = ''
            thisline = ''
            for c in range(size0):
                if c > 0:
                    thisline += ' '
                thisline += floatToString(self.get1d(c))
            res += thisline + '\n'
            res += '[torch.FloatTensor of size ' + str(size0) + ']\n'
            return res
        else:
            raise Exception("Not implemented: dims > 2")

def asTensor(myarray):
    cdef float[:] myarraymv
    if str(type(myarray)) == "<type 'numpy.ndarray'>":
        dims = len(myarray.shape)
        rows = myarray.shape[0]
        cols = myarray.shape[1]

        myarraymv = myarray.reshape(rows * cols)
        storage = FloatStorage.newWithData(myarraymv)
        tensor = FloatTensor.newWithStorage2d(storage, 0, rows, cols, cols, 1)
        return tensor
    elif isinstance(myarray, array.array):
        myarraymv = myarray
        storage = FloatStorage.newWithData(myarraymv)
        tensor = FloatTensor.newWithStorage1d(storage, 0, len(myarray), 1)
        return tensor        
    else:
        raise Exception("not implemented")

cdef extern from "nnWrapper.h":
    cdef struct lua_State
    lua_State *luaInit()
    void luaClose(lua_State *L)
    long pointerAsInt(void *ptr)

    cdef cppclass _Module:
        THFloatTensor *forward(THFloatTensor *input)
        THFloatTensor *backward(THFloatTensor *input, THFloatTensor *gradOutput)
        void zeroGradParameters()
        void updateParameters(float learningRate)
        THFloatTensor *updateOutput(THFloatTensor *input)
        THFloatTensor *updateGradInput(THFloatTensor *input, THFloatTensor *gradOutput)
        THFloatTensor *getOutput()
        THFloatTensor *getGradInput()

    cdef cppclass _Linear(_Module):
        _Linear(lua_State *L, int inputSize, int OutputSize)
        THFloatTensor *getWeight()

    cdef cppclass _LogSoftMax(_Module):
        _LogSoftMax(lua_State *L)

    cdef cppclass _Sequential(_Module):
        _Sequential(lua_State *L)
        void add(_Module *module)

    # ==== Criterions ================
    cdef cppclass _Criterion:
#        THFloatTensor *forward(THFloatTensor *input, float target)
#        THFloatTensor *backward(THFloatTensor *input, float target)
#        THFloatTensor *updateOutput(THFloatTensor *input, float target)
#        THFloatTensor *updateGradInput(THFloatTensor *input, float target)

        float forward(THFloatTensor *input, THFloatTensor *target)
        float updateOutput(THFloatTensor *input, THFloatTensor *target)
        THFloatTensor *backward(THFloatTensor *input, THFloatTensor *target)
        THFloatTensor *updateGradInput(THFloatTensor *input, THFloatTensor *target)
        float getOutput()
        THFloatTensor *getGradInput()

    cdef cppclass _MSECriterion(_Criterion):
        _MSECriterion(lua_State *L)

    cdef cppclass _ClassNLLCriterion(_Criterion):
        _ClassNLLCriterion(lua_State *L)

    # ==== trainers ====================
    cdef cppclass _Trainer:
        pass

    cdef cppclass _StochasticGradient:
        _StochasticGradient(lua_State *L, _Module *module, _Criterion *criterion)

cdef class Module(object):
    cdef _Module *native

    def forward(self, FloatTensor input):
        cdef THFloatTensor *outputC = self.native.forward(input.thFloatTensor)
        return FloatTensor.fromNative(outputC)

    def backward(self, FloatTensor input, FloatTensor gradOutput):
        cdef THFloatTensor *gradInputC = self.native.backward(input.thFloatTensor, gradOutput.thFloatTensor)
        return FloatTensor.fromNative(gradInputC)

    def zeroGradParameters(self):
        self.native.zeroGradParameters()

    def updateParameters(self, float learningRate):
        self.native.updateParameters(learningRate)

    def updateOutput(self, FloatTensor input):
        cdef THFloatTensor *outputC = self.native.updateOutput(input.thFloatTensor)
        return FloatTensor.fromNative(outputC)

    def updateGradInput(self, FloatTensor input, FloatTensor gradOutput):
        cdef THFloatTensor *gradInputC = self.native.updateGradInput(input.thFloatTensor, gradOutput.thFloatTensor)
        return FloatTensor.fromNative(gradInputC)

    @property
    def output(self):
        cdef THFloatTensor *outputC = self.native.getOutput()
        return FloatTensor.fromNative(outputC)

    @property
    def gradInput(self):
        cdef THFloatTensor *gradInputC = self.native.getGradInput()
        return FloatTensor.fromNative(gradInputC)

    # there's probably an official Torch way of doing this
    cpdef int getPrediction(self, FloatTensor output):
        cdef int prediction = 0
        cdef float maxSoFar = output[0]
        cdef float thisValue = 0
        cdef int i = 0
        for i in range(THFloatTensor_size(output.thFloatTensor, 0)):
            thisValue = THFloatTensor_get1d(output.thFloatTensor, i)
            if thisValue > maxSoFar:
                maxSoFar = thisValue
                prediction = i
        return prediction

cdef class Linear(Module):

    def __cinit__(self, Nn nn, inputSize, outputSize):
        self.native = new _Linear(nn.L, inputSize, outputSize)

    def __dealloc__(self):
        del self.native

    @property
    def weight(self):
        cdef THFloatTensor *weightC = (<_Linear *>(self.native)).getWeight()
        return FloatTensor.fromNative(weightC)

cdef class LogSoftMax(Module):
    def __cinit__(self, Nn nn):
        self.native = new _LogSoftMax(nn.L)

    def __dealloc__(self):
        del self.native

cdef class Sequential(Module):
    def __cinit__(self, Nn nn):
        self.native = new _Sequential(nn.L)

    def __dealloc__(self):
        del self.native

    def add(self, Module module):
        (<_Sequential *>(self.native)).add(module.native)
        return self

#  ==== Criterions ==========================
cdef class Criterion(object):
    cdef _Criterion *native

#    def forward(self, FloatTensor input, float target):
#        print('PyTorch.pyx Criterion.forward')
#        cdef THFloatTensor *outputC = self.native.forward(input.thFloatTensor, target)
#        return FloatTensor.fromNative(outputC)

    @property
    def output(self):
        cdef float outputC = self.native.getOutput()
        return outputC

    @property
    def gradInput(self):
        cdef THFloatTensor *gradInputC = self.native.getGradInput()
        return FloatTensor.fromNative(gradInputC)

    def forward(self, FloatTensor input, FloatTensor target):
        cdef float loss = self.native.forward(input.thFloatTensor, target.thFloatTensor)
        return loss

    def updateOutput(self, FloatTensor input, FloatTensor target):
        cdef float loss = self.native.updateOutput(input.thFloatTensor, target.thFloatTensor)
        return loss

    def backward(self, FloatTensor input, FloatTensor target):
        cdef THFloatTensor *gradInputC = self.native.backward(input.thFloatTensor, target.thFloatTensor)
        return FloatTensor.fromNative(gradInputC)

    def updateGradInput(self, FloatTensor input, FloatTensor target):
        cdef THFloatTensor *gradInputC = self.native.updateGradInput(input.thFloatTensor, target.thFloatTensor)
        return FloatTensor.fromNative(gradInputC)

cdef class MSECriterion(Criterion):
    def __cinit__(self, Nn nn):
        self.native = new _MSECriterion(nn.L)

    def __dealloc__(self):
        del self.native

cdef class ClassNLLCriterion(Criterion):
    def __cinit__(self, Nn nn):
        self.native = new _ClassNLLCriterion(nn.L)

    def __dealloc__(self):
        del self.native

# === trainers ===================
cdef class StochasticGradient(object):
    cdef _StochasticGradient *native

    def __cinit__(self, Nn nn, Module module, Criterion criterion):
        self.native = new _StochasticGradient(nn.L, module.native, criterion.native)

    def __dealloc__(self):
        del self.native

# ==== Nn ==================================
cdef class Nn(object):  # basically holds the Lua state
    cdef lua_State *L

    def __cinit__(self):
        self.L = luaInit()
    
    def __dealloc__(self):
        luaClose(self.L)

    def Linear(self, inputSize, outputSize):
        return Linear(self, inputSize, outputSize)

    def LogSoftMax(self):
        return LogSoftMax(self)

    def Sequential(self):
        return Sequential(self)

    def MSECriterion(self):
        return MSECriterion(self)

    def ClassNLLCriterion(self):
        return ClassNLLCriterion(self)

    def StochasticGradient(self, module, criterion):
        return StochasticGradient(self, module, criterion)

