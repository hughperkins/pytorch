from __future__ import print_function

import cython
cimport cython

cimport cpython.array
import array

#cdef extern from "nnWrapper.h":
#    long pointerAsInt(void *ptr)

from math import log10, floor

cimport PyTorch

# from http://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)

cdef extern from "LuaHelper.h":
    void *getGlobal(lua_State *L, const char *name1, const char *name2);
    void require(lua_State *L, const char *name)

cdef class LuaHelper(object):
    @staticmethod
    def require(name):
        require(globalState.L, name)

cdef extern from "nnWrapper.h":
#    cdef struct PyTorchState
#    PyTorchState *initPyTorchState()
#    lua_State *getL(PyTorchState *state)
    long pointerAsInt(void *ptr)
    int THFloatStorage_getRefCount(THFloatStorage *self)
    int THFloatTensor_getRefCount(THFloatTensor *self)
    void collectGarbage(lua_State *L)

cdef extern from "THRandom.h":
    cdef struct THGenerator
    void THRandom_manualSeed(THGenerator *_generator, unsigned long the_seed_)

def manualSeed(long seed):
    THRandom_manualSeed(globalState.generator, seed)

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
#        print('floatStorage.__cinit__')
        if len(args) > 0:
            raise Exception('cannot provide arguments to initializer')
        if len(kwargs) > 0:
            raise Exception('cannot provide arguments to initializer')

    @staticmethod
    cdef fromNative(THFloatStorage *storageC, retain=True):
        if retain:
            THFloatStorage_retain(storageC)
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
        return FloatStorage.fromNative(storageC, retain=False)

    cpdef long size(self):
        return THFloatStorage_size(self.thFloatStorage)

    def __dealloc__(self):
#        print('THFloatStorage.dealloc, old refcount ', THFloatStorage_getRefCount(self.thFloatStorage))
#        print('   dealloc storage: ', hex(<long>(self.thFloatStorage)))
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

    void THFloatTensor_bernoulli(THFloatTensor *self, THGenerator *_generator, double p)
    void THFloatTensor_geometric(THFloatTensor *self, THGenerator *_generator, double p)

    void THFloatTensor_uniform(THFloatTensor *self, THGenerator *_generator, double a, double b)
    void THFloatTensor_normal(THFloatTensor *self, THGenerator *_generator, double mean, double stdv)
    void THFloatTensor_exponential(THFloatTensor *self, THGenerator *_generator, double _lambda);
    void THFloatTensor_cauchy(THFloatTensor *self, THGenerator *_generator, double median, double sigma)
    void THFloatTensor_logNormal(THFloatTensor *self, THGenerator *_generator, double mean, double stdv)

    void THFloatTensor_add(THFloatTensor *r_, THFloatTensor *t, float value)
    THFloatStorage *THFloatTensor_storage(THFloatTensor *self)
    void THFloatTensor_retain(THFloatTensor *self)
    THFloatTensor *THFloatTensor_newNarrow(THFloatTensor *self, int dimension, long firstIndex, long size)

cdef class _FloatTensor(object):
#    cdef THFloatTensor *thFloatTensor

#    def __cinit__(Tensor self, THFloatTensor *tensorC = NULL):
#        self.thFloatTensor = tensorC

    def __cinit__(self, *args, **kwargs):
#        print('floatTensor.__cinit__')
        cdef THFloatStorage *storageC
        cdef long addr
        if len(kwargs) > 0:
            raise Exception('cannot provide arguments to initializer')
        if len(args) > 0:
            for arg in args:
                if not isinstance(arg, int):
                    raise Exception('cannot provide arguments to initializer')
            if len(args) == 1:
#                print('new tensor 1d length', args[0])
                self.thFloatTensor = THFloatTensor_newWithSize1d(args[0])
                storageC = THFloatTensor_storage(self.thFloatTensor)
#                if storageC == NULL:
#                    print('storageC is NULL')
#                else:
#                    print('storageC not null')
#                    addr = <long>(storageC)
#                    print('storageaddr', hex(addr))
#                    print('storageC refcount', THFloatStorage_getRefCount(storageC))
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
        cdef int refCount
#        cdef int dims
#        cdef int size
#        cdef int i
#        cdef THFloatStorage *storage
        refCount = THFloatTensor_getRefCount(self.thFloatTensor)
#        print('FloatTensor.dealloc old refcount', refCount)
#        storage = THFloatTensor_storage(self.thFloatTensor)
#        if storage == NULL:
#            print('   dealloc, storage NULL')
#        else:
#            print('   dealloc, storage ', hex(<long>(storage)))
#        dims = THFloatTensor_nDimension(self.thFloatTensor)
#        print('   dims of dealloc', dims)
#        for i in range(dims):
#            print('   size[', i, ']', THFloatTensor_size(self.thFloatTensor, i))
        if refCount < 1:
            raise Exception('Unallocated an already deallocated tensor... :-O')  # Hmmm, seems this exceptoin wont go anywhere useful... :-P
        THFloatTensor_free(self.thFloatTensor)

    @property
    def refCount(_FloatTensor self):
        return THFloatTensor_getRefCount(self.thFloatTensor)

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
    def new():
#        print('allocate tensor')
        cdef THFloatTensor *newTensorC = THFloatTensor_new()
        return _FloatTensor_fromNative(newTensorC, False)

    @staticmethod
    def newWithStorage1d(FloatStorage storage, offset, size0, stride0):
#        print('allocate tensor')
        cdef THFloatTensor *newTensorC = THFloatTensor_newWithStorage1d(storage.thFloatStorage, offset, size0, stride0)
        return _FloatTensor_fromNative(newTensorC, False)

    @staticmethod
    def newWithStorage2d(FloatStorage storage, offset, size0, stride0, size1, stride1):
#        print('allocate tensor')
        cdef THFloatTensor *newTensorC = THFloatTensor_newWithStorage2d(storage.thFloatStorage, offset, size0, stride0, size1, stride1)
        return _FloatTensor_fromNative(newTensorC, False)

    def narrow(_FloatTensor self, int dimension, long firstIndex, long size):
        cdef THFloatTensor *narrowedC = THFloatTensor_newNarrow(self.thFloatTensor, dimension, firstIndex, size)
        return _FloatTensor_fromNative(narrowedC, retain=False)

    def resize2d(_FloatTensor self, long size0, long size1):
        THFloatTensor_resize2d(self.thFloatTensor, size0, size1)
        return self

    def storage(_FloatTensor self):
        cdef THFloatStorage *storageC = THFloatTensor_storage(self.thFloatTensor)
        if storageC == NULL:
            return None
        return FloatStorage.fromNative(storageC)

    def __getitem__(_FloatTensor self, int index):
        if self.dims() == 1:
            return self.get1d(index)
        cdef THFloatTensor *res = THFloatTensor_newSelect(self.thFloatTensor, 0, index)
        return _FloatTensor_fromNative(res, False)

    def __setitem__(_FloatTensor self, int index, float value):
        if self.dims() == 1:
            self.set1d(index, value)
        else:
            raise Exception("not implemented")

    def __iadd__(_FloatTensor self, float value):
        THFloatTensor_add(self.thFloatTensor, self.thFloatTensor, value)
        return self

    def __add__(_FloatTensor self, float value):
        # assume 2d matrix for now?
        cdef _FloatTensor res = _FloatTensor.new()
#        THFloatTensor_resizeAs(cresult, self.thFloatTensor)
        THFloatTensor_add(res.thFloatTensor, self.thFloatTensor, value)
        return res

    # ========== random ===============================
    def bernoulli(_FloatTensor self, float p=0.5):
        THFloatTensor_bernoulli(self.thFloatTensor, globalState.generator, p)
        return self

    def geometric(_FloatTensor self, float p=0.5):
        THFloatTensor_geometric(self.thFloatTensor, globalState.generator, p)
        return self

    def uniform(_FloatTensor self, float a=0, float b=1):
        THFloatTensor_uniform(self.thFloatTensor, globalState.generator, a, b)
        return self

    def normal(_FloatTensor self, float mean=0, float stdv=1):
        THFloatTensor_normal(self.thFloatTensor, globalState.generator, mean, stdv)
        return self

    def exponential(_FloatTensor self, float _lambda=1):
        THFloatTensor_exponential(self.thFloatTensor, globalState.generator, _lambda)
        return self

    def cauchy(_FloatTensor self, float median=0, float sigma=1):
        THFloatTensor_cauchy(self.thFloatTensor, globalState.generator, median, sigma)
        return self

    def logNormal(_FloatTensor self, float mean=1, float stdv=2):
        THFloatTensor_logNormal(self.thFloatTensor, globalState.generator, mean, stdv)
        return self

    # ====================================

    def fill(_FloatTensor self, float value):
        THFloatTensor_fill(self.thFloatTensor, value)
        return self

    def size(_FloatTensor self):
        cdef int dims = self.dims()
        cdef _FloatTensor size
        if dims > 0:
            size = _FloatTensor(dims)
            for d in range(dims):
                size.set1d(d, THFloatTensor_size(self.thFloatTensor, d))
            return size
        else:
            return None  # not sure how to handle this yet

    def __mul__(_FloatTensor self, _FloatTensor M2):
        cdef _FloatTensor T = _FloatTensor.new()
        cdef _FloatTensor res = _FloatTensor.new()
        cdef int resRows = THFloatTensor_size(self.thFloatTensor, 0)
        cdef int resCols = THFloatTensor_size(M2.thFloatTensor, 1)
        res.resize2d(resRows, resCols)
        T.resize2d(resRows, resCols)
        THFloatTensor_addmm(res.thFloatTensor, 0, T.thFloatTensor, 1, self.thFloatTensor, M2.thFloatTensor)
        return res

    def __repr__(_FloatTensor self):
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

#class FloatTensor(_FloatTensor):
#    pass

#    @staticmethod
cdef _FloatTensor_fromNative(THFloatTensor *tensorC, retain=True):
    if retain:
        THFloatTensor_retain(tensorC)
    tensor = _FloatTensor()
    tensor.thFloatTensor = tensorC
    return tensor

def asTensor(myarray):
    cdef float[:] myarraymv
    cdef FloatStorage storage
    if str(type(myarray)) == "<type 'numpy.ndarray'>":
        dims = len(myarray.shape)
        rows = myarray.shape[0]
        cols = myarray.shape[1]

        myarraymv = myarray.reshape(rows * cols)
        storage = FloatStorage.newWithData(myarraymv)
        THFloatStorage_retain(storage.thFloatStorage) # since newWithData takes ownership
        tensor = _FloatTensor.newWithStorage2d(storage, 0, rows, cols, cols, 1)
        return tensor
    elif isinstance(myarray, array.array):
        myarraymv = myarray
        storage = FloatStorage.newWithData(myarraymv)
        THFloatStorage_retain(storage.thFloatStorage) # since newWithData takes ownership
        tensor = _FloatTensor.newWithStorage1d(storage, 0, len(myarray), 1)
        return tensor        
    else:
        raise Exception("not implemented")

cdef extern from "nnWrapper.h":
    cdef struct lua_State
    lua_State *luaInit()
    void luaClose(lua_State *L)

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

    def forward(self, _FloatTensor input):
        cdef THFloatTensor *outputC = self.native.forward(input.thFloatTensor)
        return _FloatTensor_fromNative(outputC)

    def backward(self, _FloatTensor input, _FloatTensor gradOutput):
        cdef THFloatTensor *gradInputC = self.native.backward(input.thFloatTensor, gradOutput.thFloatTensor)
        return _FloatTensor_fromNative(gradInputC)

    def zeroGradParameters(self):
        self.native.zeroGradParameters()

    def updateParameters(self, float learningRate):
        self.native.updateParameters(learningRate)

    def updateOutput(self, _FloatTensor input):
        cdef THFloatTensor *outputC = self.native.updateOutput(input.thFloatTensor)
        return _FloatTensor_fromNative(outputC)

    def updateGradInput(self, _FloatTensor input, _FloatTensor gradOutput):
        cdef THFloatTensor *gradInputC = self.native.updateGradInput(input.thFloatTensor, gradOutput.thFloatTensor)
        return _FloatTensor_fromNative(gradInputC)

    @property
    def output(self):
        cdef THFloatTensor *outputC = self.native.getOutput()
        output = _FloatTensor_fromNative(outputC)
        return output

    @property
    def gradInput(self):
        cdef THFloatTensor *gradInputC = self.native.getGradInput()
        return _FloatTensor_fromNative(gradInputC)

    # there's probably an official Torch way of doing this
    cpdef int getPrediction(self, _FloatTensor output):
        cdef int prediction = 0
        cdef float maxSoFar = output[0]
        cdef float thisValue = 0
        cdef int i = 0
        for i in range(THFloatTensor_size(output.thFloatTensor, 0)):
            thisValue = THFloatTensor_get1d(output.thFloatTensor, i)
            if thisValue > maxSoFar:
                maxSoFar = thisValue
                prediction = i
        return prediction + 1  # As Karpathy would say: "sigh lua" :-P

cdef class Linear(Module):

    def __cinit__(self, Nn nn, inputSize, outputSize):
        self.native = new _Linear(nn.L, inputSize, outputSize)

    def __dealloc__(self):
        del self.native

    @property
    def weight(self):
        cdef THFloatTensor *weightC = (<_Linear *>(self.native)).getWeight()
        return _FloatTensor_fromNative(weightC)

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

    @property
    def output(self):
        cdef float outputC = self.native.getOutput()
        return outputC

    @property
    def gradInput(self):
        cdef THFloatTensor *gradInputC = self.native.getGradInput()
        return _FloatTensor_fromNative(gradInputC)

    def forward(self, _FloatTensor input, _FloatTensor target):
        cdef float loss = self.native.forward(input.thFloatTensor, target.thFloatTensor)
        return loss

    def updateOutput(self, _FloatTensor input, _FloatTensor target):
        cdef float loss = self.native.updateOutput(input.thFloatTensor, target.thFloatTensor)
        return loss

    def backward(self, _FloatTensor input, _FloatTensor target):
        cdef THFloatTensor *gradInputC = self.native.backward(input.thFloatTensor, target.thFloatTensor)
        return _FloatTensor_fromNative(gradInputC)

    def updateGradInput(self, _FloatTensor input, _FloatTensor target):
        cdef THFloatTensor *gradInputC = self.native.updateGradInput(input.thFloatTensor, target.thFloatTensor)
        return _FloatTensor_fromNative(gradInputC)

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
#        self.L = luaInit()
        self.L = globalState.L
#        self.L = globalState.getL()
    
    def __dealloc__(self):
        pass
#        luaClose(self.L)

    def collectgarbage(self):
        collectGarbage(self.L)

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

cdef class GlobalState(object):
#    cdef PyTorchState *state
#    cdef lua_State *L
#    cdef THGenerator *generator

    def __cinit__(GlobalState self):
        print('GlobalState.__cinit__')
#        self.state = initPyTorchState();

    def __dealloc__(self):
        print('GlobalState.__dealloc__')

#    cdef lua_State *getL(self):  # this is mostly a migration path, we will push this downwards, and out of htis layer
#        return getL(self.state)

cdef GlobalState globalState

def getGlobalState():
    global globalState
    return globalState

def init():
    global globalState
    print('initializing PyTorch...')
    globalState = GlobalState()
    globalState.L = luaInit()
    globalState.generator = <THGenerator *>(getGlobal(globalState.L, 'torch', '_gen'))
    print('generator null:', globalState.generator == NULL)
    print(' ... PyTorch initialized')

init()

from floattensor import FloatTensor


