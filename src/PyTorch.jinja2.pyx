# {{header1}}
# {{header2}}

from __future__ import print_function
import numbers
import cython
cimport cython

cimport cpython.array
import array

from math import log10, floor

cimport Storage
import Storage
from lua cimport *
from nnWrapper cimport *
cimport PyTorch

{% set types = {
    'Long': {'real': 'long'},
    'Float': {'real': 'float'}, 
    'Double': {'real': 'double'},
    'Byte': {'real': 'unsigned char'}
}
%}

#define real unsigned char
#define accreal long
#define Real Byte
#define TH_REAL_IS_BYTE

# from http://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)

cdef extern from "THRandom.h":
    cdef struct THGenerator
    void THRandom_manualSeed(THGenerator *_generator, unsigned long the_seed_)

def manualSeed(long seed):
    THRandom_manualSeed(globalState.generator, seed)

cdef floatToString(float floatValue):
    return '%.6g'% floatValue

{% for Real in types %}
{% set real = types[Real]['real'] %}

{{Real}}Storage = Storage.{{Real}}Storage

cdef extern from "THTensor.h":
    cdef struct TH{{Real}}Tensor
    TH{{Real}}Tensor *TH{{Real}}Tensor_new()
    TH{{Real}}Tensor *TH{{Real}}Tensor_newWithSize1d(long size0)
    TH{{Real}}Tensor *TH{{Real}}Tensor_newWithSize2d(long size0, long size1)
    TH{{Real}}Tensor* TH{{Real}}Tensor_newWithStorage1d(Storage.TH{{Real}}Storage *storage, long storageOffset, long size0, long stride0)
    TH{{Real}}Tensor* TH{{Real}}Tensor_newWithStorage2d(Storage.TH{{Real}}Storage *storage, long storageOffset, long size0, long stride0, long size1, long stride1)
    void TH{{Real}}Tensor_retain(TH{{Real}}Tensor *self)
    void TH{{Real}}Tensor_free(TH{{Real}}Tensor *self)

    int TH{{Real}}Tensor_nDimension(TH{{Real}}Tensor *tensor)
    void TH{{Real}}Tensor_resizeAs(TH{{Real}}Tensor *self, THFloatTensor *model)
    void TH{{Real}}Tensor_resize1d(TH{{Real}}Tensor *self, long size0)
    void TH{{Real}}Tensor_resize2d(TH{{Real}}Tensor *self, long size0, long size1)
    void TH{{Real}}Tensor_resize3d(TH{{Real}}Tensor *self, long size0, long size1, long size2)
    void TH{{Real}}Tensor_resize4d(TH{{Real}}Tensor *self, long size0, long size1, long size2, long size3)
    long TH{{Real}}Tensor_size(const TH{{Real}}Tensor *self, int dim)
    long TH{{Real}}Tensor_nElement(TH{{Real}}Tensor *self)
    long TH{{Real}}Tensor_stride(const TH{{Real}}Tensor *self, int dim)

    void TH{{Real}}Tensor_set1d(const TH{{Real}}Tensor *tensor, long x0, float value)
    void TH{{Real}}Tensor_set2d(const TH{{Real}}Tensor *tensor, long x0, long x1, float value)
    {{real}} TH{{Real}}Tensor_get1d(const TH{{Real}}Tensor *tensor, long x0)
    {{real}} TH{{Real}}Tensor_get2d(const TH{{Real}}Tensor *tensor, long x0, long x1)

    void TH{{Real}}Tensor_fill(TH{{Real}}Tensor *self, {{real}} value)
    TH{{Real}}Tensor *TH{{Real}}Tensor_newSelect(TH{{Real}}Tensor *self, int dimension, int sliceIndex)
    TH{{Real}}Tensor *TH{{Real}}Tensor_newNarrow(TH{{Real}}Tensor *self, int dimension, long firstIndex, long size)
    Storage.TH{{Real}}Storage *TH{{Real}}Tensor_storage(TH{{Real}}Tensor *self)

    void TH{{Real}}Tensor_add(TH{{Real}}Tensor *r_, TH{{Real}}Tensor *t, {{real}} value)
    void TH{{Real}}Tensor_div(TH{{Real}}Tensor *r_, TH{{Real}}Tensor *t, {{real}} value)
    void TH{{Real}}Tensor_mul(TH{{Real}}Tensor *r_, TH{{Real}}Tensor *t, {{real}} value)
    void TH{{Real}}Tensor_add(TH{{Real}}Tensor *tensorSelf, TH{{Real}}Tensor *tensorOne, {{real}} value)

    void TH{{Real}}Tensor_geometric(TH{{Real}}Tensor *self, THGenerator *_generator, double p)
    void TH{{Real}}Tensor_bernoulli(TH{{Real}}Tensor *self, THGenerator *_generator, double p)

    {% if Real in ['Float', 'Double'] %}
    void TH{{Real}}Tensor_addmm(TH{{Real}}Tensor *tensorSelf, double beta, TH{{Real}}Tensor *tensorOne, double alpha, TH{{Real}}Tensor *mat1, TH{{Real}}Tensor *mat2)

    void TH{{Real}}Tensor_uniform(TH{{Real}}Tensor *self, THGenerator *_generator, double a, double b)
    void TH{{Real}}Tensor_normal(TH{{Real}}Tensor *self, THGenerator *_generator, double mean, double stdv)
    void TH{{Real}}Tensor_exponential(TH{{Real}}Tensor *self, THGenerator *_generator, double _lambda);
    void TH{{Real}}Tensor_cauchy(TH{{Real}}Tensor *self, THGenerator *_generator, double median, double sigma)
    void TH{{Real}}Tensor_logNormal(TH{{Real}}Tensor *self, THGenerator *_generator, double mean, double stdv)
    {% endif %}
{% endfor %}

{% for Real in types %}
{% set real = types[Real]['real'] %}
cdef class _{{Real}}Tensor(object):
    # properties are in the PyTorch.pxd file

#    def __cinit__(Tensor self, THFloatTensor *tensorC = NULL):
#        self.thFloatTensor = tensorC

    def __cinit__(self, *args, _allocate=True):
#        print('{{Real}}Tensor.__cinit__')
#        cdef TH{{Real}}Storage *storageC
#        cdef long addr
#        if len(kwargs) > 0:
#            raise Exception('cannot provide arguments to initializer')
        if _allocate:
#            if len(args) == 1 and isinstance(args[0], _LongTensor):  # it's a size tensor
#                self.thFloatTensor = THFloatTensor_new()
            for arg in args:
                if not isinstance(arg, int):
                    raise Exception('cannot provide arguments to initializer')
            if len(args) == 0:
#                print('no args, calling TH{{Real}}Tensor_new()')
                self.th{{Real}}Tensor = TH{{Real}}Tensor_new()
            elif len(args) == 1:
#                print('new tensor 1d length', args[0])
                self.th{{Real}}Tensor = TH{{Real}}Tensor_newWithSize1d(args[0])
#                storageC = THFloatTensor_storage(self.thFloatTensor)
#                if storageC == NULL:
#                    print('storageC is NULL')
#                else:
#                    print('storageC not null')
#                    addr = <long>(storageC)
#                    print('storageaddr', hex(addr))
#                    print('storageC refcount', THFloatStorage_getRefCount(storageC))
            elif len(args) == 2:
                self.th{{Real}}Tensor = TH{{Real}}Tensor_newWithSize2d(args[0], args[1])
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
        refCount = TH{{Real}}Tensor_getRefCount(self.th{{Real}}Tensor)
#        print('{{Real}}Tensor.dealloc old refcount', refCount)
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
        TH{{Real}}Tensor_free(self.th{{Real}}Tensor)

    def nElement(_{{Real}}Tensor self):
        return TH{{Real}}Tensor_nElement(self.th{{Real}}Tensor)

    @property
    def refCount(_{{Real}}Tensor self):
        return TH{{Real}}Tensor_getRefCount(self.th{{Real}}Tensor)

    cpdef int dims(self):
        return TH{{Real}}Tensor_nDimension(self.th{{Real}}Tensor)

    cpdef set1d(self, int x0, {{real}} value):
        TH{{Real}}Tensor_set1d(self.th{{Real}}Tensor, x0, value)

    cpdef set2d(self, int x0, int x1, {{real}} value):
        TH{{Real}}Tensor_set2d(self.th{{Real}}Tensor, x0, x1, value)

    cpdef {{real}} get1d(self, int x0):
        return TH{{Real}}Tensor_get1d(self.th{{Real}}Tensor, x0)

    cpdef {{real}} get2d(self, int x0, int x1):
        return TH{{Real}}Tensor_get2d(self.th{{Real}}Tensor, x0, x1)

    def __repr__(_{{Real}}Tensor self):
        # assume 2d matrix for now
        cdef int size0
        cdef int size1
        dims = self.dims()
        if dims == 0:
            return '[torch.{{Real}}Tensor with no dimension]\n'
        elif dims == 2:
            size0 = TH{{Real}}Tensor_size(self.th{{Real}}Tensor, 0)
            size1 = TH{{Real}}Tensor_size(self.th{{Real}}Tensor, 1)
            res = ''
            for r in range(size0):
                thisline = ''
                for c in range(size1):
                    if c > 0:
                        thisline += ' '
                    {% if Real in ['Float'] %}
                    thisline += floatToString(self.get2d(r,c),)
                    {% else %}
                    thisline += str(self.get2d(r,c),)
                    {% endif %}
                res += thisline + '\n'
            res += '[torch.{{Real}}Tensor of size ' + ('%.0f' % size0) + 'x' + str(size1) + ']\n'
            return res
        elif dims == 1:
            size0 = TH{{Real}}Tensor_size(self.th{{Real}}Tensor, 0)
            res = ''
            thisline = ''
            for c in range(size0):
                if c > 0:
                    thisline += ' '
                {% if Real in ['Float'] %}
                thisline += floatToString(self.get1d(c))
                {% else %}
                thisline += str(self.get1d(c))
                {% endif %}
            res += thisline + '\n'
            res += '[torch.{{Real}}Tensor of size ' + str(size0) + ']\n'
            return res
        else:
            raise Exception("Not implemented: dims > 2")

    def __getitem__(_{{Real}}Tensor self, int index):
        if self.dims() == 1:
            return self.get1d(index)
        cdef TH{{Real}}Tensor *res = TH{{Real}}Tensor_newSelect(self.th{{Real}}Tensor, 0, index)
        return _{{Real}}Tensor_fromNative(res, False)

    def __setitem__(_{{Real}}Tensor self, int index, {{real}} value):
        if self.dims() == 1:
            self.set1d(index, value)
        else:
            raise Exception("not implemented")

    def fill(_{{Real}}Tensor self, {{real}} value):
        TH{{Real}}Tensor_fill(self.th{{Real}}Tensor, value)
        return self

    def size(_{{Real}}Tensor self):
        cdef int dims = self.dims()
        cdef _LongTensor size
        if dims > 0:
            size = _LongTensor(dims)
            for d in range(dims):
                size.set1d(d, TH{{Real}}Tensor_size(self.th{{Real}}Tensor, d))
            return size
        else:
            return None  # not sure how to handle this yet

    @staticmethod
    def new():
#        print('allocate tensor')
        return _{{Real}}Tensor()
#        return _FloatTensor_fromNative(newTensorC, False)

    def narrow(_{{Real}}Tensor self, int dimension, long firstIndex, long size):
        cdef TH{{Real}}Tensor *narrowedC = TH{{Real}}Tensor_newNarrow(self.th{{Real}}Tensor, dimension, firstIndex, size)
        return _{{Real}}Tensor_fromNative(narrowedC, retain=False)

    def resize1d(_{{Real}}Tensor self, int size0):
        TH{{Real}}Tensor_resize1d(self.th{{Real}}Tensor, size0)
        return self

    def resize2d(_{{Real}}Tensor self, int size0, int size1):
        TH{{Real}}Tensor_resize2d(self.th{{Real}}Tensor, size0, size1)
        return self

    def resize3d(_{{Real}}Tensor self, int size0, int size1, int size2):
        TH{{Real}}Tensor_resize3d(self.th{{Real}}Tensor, size0, size1, size2)
        return self

    def resize4d(_{{Real}}Tensor self, int size0, int size1, int size2, int size3):
        TH{{Real}}Tensor_resize4d(self.th{{Real}}Tensor, size0, size1, size2, size3)
        return self

    def resize(_{{Real}}Tensor self, _LongTensor size):
#        print('_FloatTensor.resize size:', size)
        if size.dims() == 0:
            return self
        cdef int dims = size.size()[0]
#        print('_FloatTensor.resize dims:', dims)
        if dims == 1:
            TH{{Real}}Tensor_resize1d(self.th{{Real}}Tensor, size[0])
        elif dims == 2:
            TH{{Real}}Tensor_resize2d(self.th{{Real}}Tensor, size[0], size[1])
        elif dims == 3:
            TH{{Real}}Tensor_resize3d(self.th{{Real}}Tensor, size[0], size[1], size[2])
        elif dims == 4:
            TH{{Real}}Tensor_resize4d(self.th{{Real}}Tensor, size[0], size[1], size[2], size[3])
        else:
            raise Exception('Not implemented for dims=' + str(dims))
        return self

    @staticmethod
    def newWithStorage1d(Storage.{{Real}}Storage storage, offset, size0, stride0):
#        print('allocate tensor')
        cdef TH{{Real}}Tensor *newTensorC = TH{{Real}}Tensor_newWithStorage1d(storage.th{{Real}}Storage, offset, size0, stride0)
        return _{{Real}}Tensor_fromNative(newTensorC, False)

    @staticmethod
    def newWithStorage2d(Storage.{{Real}}Storage storage, offset, size0, stride0, size1, stride1):
#        print('allocate tensor')
        cdef TH{{Real}}Tensor *newTensorC = TH{{Real}}Tensor_newWithStorage2d(storage.th{{Real}}Storage, offset, size0, stride0, size1, stride1)
        return _{{Real}}Tensor_fromNative(newTensorC, False)

    def storage(_{{Real}}Tensor self):
        cdef Storage.TH{{Real}}Storage *storageC = TH{{Real}}Tensor_storage(self.th{{Real}}Tensor)
        if storageC == NULL:
            return None
        return Storage.{{Real}}Storage_fromNative(storageC)

    def __add__(_{{Real}}Tensor self, {{real}} value):
        # assume 2d matrix for now?
        cdef _{{Real}}Tensor res = _{{Real}}Tensor.new()
#        THFloatTensor_resizeAs(cresult, self.thFloatTensor)
        TH{{Real}}Tensor_add(res.th{{Real}}Tensor, self.th{{Real}}Tensor, value)
        return res

    def __sub__(_{{Real}}Tensor self, {{real}} value):
        # assume 2d matrix for now?
        cdef _{{Real}}Tensor res = _{{Real}}Tensor.new()
#        THFloatTensor_resizeAs(cresult, self.thFloatTensor)
        TH{{Real}}Tensor_add(res.th{{Real}}Tensor, self.th{{Real}}Tensor, -value)
        return res

    def __div__(_{{Real}}Tensor self, {{real}} value):
        # assume 2d matrix for now?
        cdef _{{Real}}Tensor res = _{{Real}}Tensor.new()
#        THFloatTensor_resizeAs(cresult, self.thFloatTensor)
        TH{{Real}}Tensor_div(res.th{{Real}}Tensor, self.th{{Real}}Tensor, value)
        return res

    def __iadd__(_{{Real}}Tensor self, {{real}} value):
        TH{{Real}}Tensor_add(self.th{{Real}}Tensor, self.th{{Real}}Tensor, value)
        return self

    def __isub__(_{{Real}}Tensor self, {{real}} value):
        TH{{Real}}Tensor_add(self.th{{Real}}Tensor, self.th{{Real}}Tensor, -value)
        return self

    def __idiv__(_{{Real}}Tensor self, {{real}} value):
        TH{{Real}}Tensor_div(self.th{{Real}}Tensor, self.th{{Real}}Tensor, value)
        return self

    def __imul__(_{{Real}}Tensor self, {{real}} value):
        TH{{Real}}Tensor_mul(self.th{{Real}}Tensor, self.th{{Real}}Tensor, value)
        return self

#    def __mul__(_{{Real}}Tensor self, _{{Real}}Tensor M2):
    def __mul__(_{{Real}}Tensor self, second):
        cdef _{{Real}}Tensor M2
        cdef _{{Real}}Tensor T
        cdef _{{Real}}Tensor res
        cdef int resRows
        cdef int resCols

        res = _{{Real}}Tensor.new()
        if isinstance(second, numbers.Number):
            TH{{Real}}Tensor_mul(res.th{{Real}}Tensor, self.th{{Real}}Tensor, second)
            return res
        else:
        {% if Real in ['Float', 'Double'] %}
            M2 = second
            T = _{{Real}}Tensor.new()
            resRows = TH{{Real}}Tensor_size(self.th{{Real}}Tensor, 0)
            resCols = TH{{Real}}Tensor_size(M2.th{{Real}}Tensor, 1)
            res.resize2d(resRows, resCols)
            T.resize2d(resRows, resCols)
            TH{{Real}}Tensor_addmm(res.th{{Real}}Tensor, 0, T.th{{Real}}Tensor, 1, self.th{{Real}}Tensor, M2.th{{Real}}Tensor)
            return res
        {% else %}
            raise Exception('Invalid arg type for second: ' + str(type(second)))
        {% endif %}

    # ========== random ===============================

    def bernoulli(_{{Real}}Tensor self, float p=0.5):
        TH{{Real}}Tensor_bernoulli(self.th{{Real}}Tensor, globalState.generator, p)
        return self

    def geometric(_{{Real}}Tensor self, float p=0.5):
        TH{{Real}}Tensor_geometric(self.th{{Real}}Tensor, globalState.generator, p)
        return self

{% if Real in ['Float', 'Double'] %}
    def normal(_{{Real}}Tensor self, {{real}} mean=0, {{real}} stdv=1):
        TH{{Real}}Tensor_normal(self.th{{Real}}Tensor, globalState.generator, mean, stdv)
        return self

    def exponential(_{{Real}}Tensor self, {{real}} _lambda=1):
        TH{{Real}}Tensor_exponential(self.th{{Real}}Tensor, globalState.generator, _lambda)
        return self

    def cauchy(_{{Real}}Tensor self, {{real}} median=0, {{real}} sigma=1):
        TH{{Real}}Tensor_cauchy(self.th{{Real}}Tensor, globalState.generator, median, sigma)
        return self

    def logNormal(_{{Real}}Tensor self, {{real}} mean=1, {{real}} stdv=2):
        TH{{Real}}Tensor_logNormal(self.th{{Real}}Tensor, globalState.generator, mean, stdv)
        return self

    def uniform(_{{Real}}Tensor self, {{real}} a=0, {{real}} b=1):
        TH{{Real}}Tensor_uniform(self.th{{Real}}Tensor, globalState.generator, a, b)
        return self
{% endif %}

#    @staticmethod
cdef _{{Real}}Tensor_fromNative(TH{{Real}}Tensor *tensorC, retain=True):
    if retain:
        TH{{Real}}Tensor_retain(tensorC)
    tensor = _{{Real}}Tensor(_allocate=False)
    tensor.th{{Real}}Tensor = tensorC
    return tensor

{% endfor %}

def asFloatTensor(myarray):
    cdef float[:] myarraymv
    cdef Storage.FloatStorage storage
    if str(type(myarray)) == "<type 'numpy.ndarray'>":
        dims = len(myarray.shape)
        rows = myarray.shape[0]
        cols = myarray.shape[1]

        myarraymv = myarray.reshape(rows * cols)
        storage = Storage.FloatStorage.newWithData(myarraymv)
        Storage.THFloatStorage_retain(storage.thFloatStorage) # since newWithData takes ownership
        tensor = _FloatTensor.newWithStorage2d(storage, 0, rows, cols, cols, 1)
        return tensor
    elif isinstance(myarray, array.array):
        myarraymv = myarray
        storage = Storage.FloatStorage.newWithData(myarraymv)
        Storage.THFloatStorage_retain(storage.thFloatStorage) # since newWithData takes ownership
        tensor = _FloatTensor.newWithStorage1d(storage, 0, len(myarray), 1)
        return tensor        
    else:
        raise Exception("not implemented")

def asDoubleTensor(myarray):
    cdef double[:] myarraymv
    cdef Storage.DoubleStorage storage
    if str(type(myarray)) == "<type 'numpy.ndarray'>":
        dims = len(myarray.shape)
        rows = myarray.shape[0]
        cols = myarray.shape[1]

        myarraymv = myarray.reshape(rows * cols)
        storage = Storage.DoubleStorage.newWithData(myarraymv)
        Storage.THDoubleStorage_retain(storage.thDoubleStorage) # since newWithData takes ownership
        tensor = _DoubleTensor.newWithStorage2d(storage, 0, rows, cols, cols, 1)
        return tensor
    elif isinstance(myarray, array.array):
        myarraymv = myarray
        storage = Storage.DoubleStorage.newWithData(myarraymv)
        Storage.THDoubleStorage_retain(storage.thDoubleStorage) # since newWithData takes ownership
        tensor = _DoubleTensor.newWithStorage1d(storage, 0, len(myarray), 1)
        return tensor        
    else:
        raise Exception("not implemented")

cdef class GlobalState(object):
    # properties are in the PyTorch.pxd file

    def __cinit__(GlobalState self):
        pass
#        print('GlobalState.__cinit__')

    def __dealloc__(self):
        pass
#        print('GlobalState.__dealloc__')

    def getLua(self):
        return LuaState_fromNative(self.L)

{% for Real in types %}
{% if Real in ['Double', 'Float'] %}
def _pop{{Real}}Tensor():
    global globalState
    cdef TH{{Real}}Tensor *tensorC = pop{{Real}}Tensor(globalState.L)
    return _{{Real}}Tensor_fromNative(tensorC)

def _push{{Real}}Tensor(_{{Real}}Tensor tensor):
    global globalState
    push{{Real}}Tensor(globalState.L, tensor.th{{Real}}Tensor)
{% endif %}
{% endfor %}

# there's probably an official Torch way of doing this
{% for Real in types %}
{% set real = types[Real]['real'] %}
{% if Real in ['Double', 'Float'] %}
cpdef int get{{Real}}Prediction(_{{Real}}Tensor output):
    cdef int prediction = 0
    cdef {{real}} maxSoFar = output[0]
    cdef {{real}} thisValue = 0
    cdef int i = 0
    for i in range(TH{{Real}}Tensor_size(output.th{{Real}}Tensor, 0)):
        thisValue = TH{{Real}}Tensor_get1d(output.th{{Real}}Tensor, i)
        if thisValue > maxSoFar:
            maxSoFar = thisValue
            prediction = i
    return prediction + 1
{% endif %}
{% endfor %}

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

from floattensor import *

# ==== Nn ==================================
cdef class Nn(object):  # just used to provide the `nn.` syntax
    def collectgarbage(self):
        collectGarbage(globalState.L)

#    def Linear(self, inputSize, outputSize):
#        return Linear(inputSize, outputSize)

