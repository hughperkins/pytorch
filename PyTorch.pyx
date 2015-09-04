# Original file is: PyTorch.jinja2.pyx
# If you are looking at PyTorch.pyx, it is a generated file, dont edit it directly ;-)

from __future__ import print_function

import cython
cimport cython

cimport cpython.array
import array

from math import log10, floor

cimport PyTorch



# from http://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)

cdef extern from "LuaHelper.h":
    void *getGlobal(lua_State *L, const char *name1, const char *name2);
    void require(lua_State *L, const char *name)
    THFloatTensor *popFloatTensor(lua_State *L)
    void pushFloatTensor(lua_State *L, THFloatTensor *tensor)

cdef class LuaHelper(object):
    @staticmethod
    def require(name):
        require(globalState.L, name)

cdef extern from "lua_externc.h":
    struct lua_State
    # int LUA_REGISTRYINDEX
    void lua_pushnumber(lua_State *L, float number)
    float lua_tonumber(lua_State *L, int index)
    void lua_pushstring(lua_State *L, const char *value)
    const char *lua_tostring(lua_State *L, int index)
    void lua_call(lua_State *L, int argsIn, int argsOut)
    void lua_remove(lua_State *L, int index)
    void lua_insert(lua_State *L, int index)
    void lua_getglobal(lua_State *L, const char *name)
    void lua_setglobal(lua_State *L, const char *name)
    void lua_settable(lua_State *L, int index)
    void lua_gettable(lua_State *L, int index)
    void lua_getfield(lua_State *L, int index, const char *name)
    void lua_pushnil(lua_State *L)
    void lua_pushvalue(lua_State *L, int index)
    int lua_next(lua_State *L, int index)
    int lua_gettop(lua_State *L)
    int lua_isuserdata(lua_State *L, int index)

cdef class LuaState(object):
    cdef lua_State *L

    def __cinit__(self):
#        print('LuaState.__cinit__')
        self.L = luaInit()

    def __dealloc__(self):
#        print('LuaState.__dealloc__')
        pass

    def insert(self, int index):
        lua_insert(self.L, index)

    def remove(self, int index):
        lua_remove(self.L, index)

    def pushNumber(self, float number):
        lua_pushnumber(self.L, number)

    def pushString(self, mystring):
        lua_pushstring(self.L, mystring)

    def toString(self, int index):
        cdef bytes py_string = lua_tostring(self.L, index)
        return py_string

    def toNumber(self, int index):
        return lua_tonumber(self.L, index)

    def getGlobal(self, name):
        lua_getglobal(self.L, name)

    def setGlobal(self, name):
        lua_setglobal(self.L, name)

    def pushNil(self):
        lua_pushnil(self.L)

    def pushValue(self, int index):
        lua_pushvalue(self.L, index)

    def call(self, int numIn, int numOut):
        lua_call(self.L, numIn, numOut)

    def getField(self, int index, name):
        lua_getfield(self.L, index, name)

    def setRegistry(self):
        lua_settable(self.L, LUA_REGISTRYINDEX)

    def getRegistry(self):
        lua_gettable(self.L, LUA_REGISTRYINDEX)

    def next(self, int index):
        return lua_next(self.L, index)

    def getTop(self):
        return lua_gettop(self.L)

    def isUserData(self, int index):
        return lua_isuserdata(self.L, index)

cdef LuaState_fromNative(lua_State *L):
    cdef LuaState luaState = LuaState()
    luaState.L = L
    return luaState

cdef extern from "nnWrapper.h":
    long pointerAsInt(void *ptr)
    void collectGarbage(lua_State *L)


cdef extern from "nnWrapper.h":
    int THFloatStorage_getRefCount(THFloatStorage *self)
    int THFloatTensor_getRefCount(THFloatTensor *self)

cdef extern from "nnWrapper.h":
    int THLongStorage_getRefCount(THLongStorage *self)
    int THLongTensor_getRefCount(THLongTensor *self)


cdef extern from "THRandom.h":
    cdef struct THGenerator
    void THRandom_manualSeed(THGenerator *_generator, unsigned long the_seed_)

def manualSeed(long seed):
    THRandom_manualSeed(globalState.generator, seed)



cdef extern from "THStorage.h":
    cdef struct THFloatStorage

cdef extern from "THStorage.h":
    cdef struct THLongStorage


cdef extern from "THStorage.h":
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
    THFloatTensor *THFloatTensor_new()
    THFloatTensor *THFloatTensor_newWithSize1d(long size0)
    THFloatTensor *THFloatTensor_newWithSize2d(long size0, long size1)
    void THFloatTensor_free(THFloatTensor *self)
    int THFloatTensor_nDimension(THFloatTensor *tensor)
    THFloatTensor *THFloatTensor_newSelect(THFloatTensor *self, int dimension, int sliceIndex)
    void THFloatTensor_resizeAs(THFloatTensor *self, THFloatTensor *model)
    void THFloatTensor_resize1d(THFloatTensor *self, long size0)
    void THFloatTensor_resize2d(THFloatTensor *self, long size0, long size1)
    void THFloatTensor_resize3d(THFloatTensor *self, long size0, long size1, long size2)
    void THFloatTensor_resize4d(THFloatTensor *self, long size0, long size1, long size2, long size3)
    long THFloatTensor_size(const THFloatTensor *self, int dim)
    long THFloatTensor_nElement(THFloatTensor *self)
    void THFloatTensor_retain(THFloatTensor *self)
    void THFloatTensor_geometric(THFloatTensor *self, THGenerator *_generator, double p)
    void THFloatTensor_set1d(const THFloatTensor *tensor, long x0, float value)
    void THFloatTensor_set2d(const THFloatTensor *tensor, long x0, long x1, float value)
    float THFloatTensor_get1d(const THFloatTensor *tensor, long x0)
    float THFloatTensor_get2d(const THFloatTensor *tensor, long x0, long x1)
    long THFloatTensor_stride(const THFloatTensor *self, int dim)
    void THFloatTensor_fill(THFloatTensor *self, float value)
    void THFloatTensor_add(THFloatTensor *r_, THFloatTensor *t, float value)


cdef extern from "THTensor.h":
    cdef struct THLongTensor
    THLongTensor *THLongTensor_new()
    THLongTensor *THLongTensor_newWithSize1d(long size0)
    THLongTensor *THLongTensor_newWithSize2d(long size0, long size1)
    void THLongTensor_free(THLongTensor *self)
    int THLongTensor_nDimension(THLongTensor *tensor)
    THLongTensor *THLongTensor_newSelect(THLongTensor *self, int dimension, int sliceIndex)
    void THLongTensor_resizeAs(THLongTensor *self, THFloatTensor *model)
    void THLongTensor_resize1d(THLongTensor *self, long size0)
    void THLongTensor_resize2d(THLongTensor *self, long size0, long size1)
    void THLongTensor_resize3d(THLongTensor *self, long size0, long size1, long size2)
    void THLongTensor_resize4d(THLongTensor *self, long size0, long size1, long size2, long size3)
    long THLongTensor_size(const THLongTensor *self, int dim)
    long THLongTensor_nElement(THLongTensor *self)
    void THLongTensor_retain(THLongTensor *self)
    void THLongTensor_geometric(THLongTensor *self, THGenerator *_generator, double p)
    void THLongTensor_set1d(const THLongTensor *tensor, long x0, float value)
    void THLongTensor_set2d(const THLongTensor *tensor, long x0, long x1, float value)
    long THLongTensor_get1d(const THLongTensor *tensor, long x0)
    long THLongTensor_get2d(const THLongTensor *tensor, long x0, long x1)
    long THLongTensor_stride(const THLongTensor *self, int dim)
    void THLongTensor_fill(THLongTensor *self, long value)
    void THLongTensor_add(THLongTensor *r_, THLongTensor *t, long value)


cdef extern from "THTensor.h":
    THFloatTensor* THFloatTensor_newWithStorage1d(THFloatStorage *storage, long storageOffset, long size0, long stride0)
    THFloatTensor* THFloatTensor_newWithStorage2d(THFloatStorage *storage, long storageOffset, long size0, long stride0, long size1, long stride1)
    void THFloatTensor_add(THFloatTensor *tensorSelf, THFloatTensor *tensorOne, float value)
    void THFloatTensor_addmm(THFloatTensor *tensorSelf, float beta, THFloatTensor *tensorOne, float alpha, THFloatTensor *mat1, THFloatTensor *mat2)

    void THFloatTensor_bernoulli(THFloatTensor *self, THGenerator *_generator, double p)

    void THFloatTensor_uniform(THFloatTensor *self, THGenerator *_generator, double a, double b)
    void THFloatTensor_normal(THFloatTensor *self, THGenerator *_generator, double mean, double stdv)
    void THFloatTensor_exponential(THFloatTensor *self, THGenerator *_generator, double _lambda);
    void THFloatTensor_cauchy(THFloatTensor *self, THGenerator *_generator, double median, double sigma)
    void THFloatTensor_logNormal(THFloatTensor *self, THGenerator *_generator, double mean, double stdv)

    THFloatStorage *THFloatTensor_storage(THFloatTensor *self)
    THFloatTensor *THFloatTensor_newNarrow(THFloatTensor *self, int dimension, long firstIndex, long size)



cdef class _FloatTensor(object):
    # properties are in the PyTorch.pxd file

#    def __cinit__(Tensor self, THFloatTensor *tensorC = NULL):
#        self.thFloatTensor = tensorC

    def __cinit__(self, *args, _allocate=True):
        print('FloatTensor.__cinit__')
#        cdef THFloatStorage *storageC
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
                print('no args, calling THFloatTensor_new()')
                self.thFloatTensor = THFloatTensor_new()
            elif len(args) == 1:
#                print('new tensor 1d length', args[0])
                self.thFloatTensor = THFloatTensor_newWithSize1d(args[0])
#                storageC = THFloatTensor_storage(self.thFloatTensor)
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
        print('FloatTensor.dealloc old refcount', refCount)
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

    def nElement(_FloatTensor self):
        return THFloatTensor_nElement(self.thFloatTensor)

    @property
    def refCount(_FloatTensor self):
        return THFloatTensor_getRefCount(self.thFloatTensor)

    def geometric(_FloatTensor self, float p=0.5):
        THFloatTensor_geometric(self.thFloatTensor, globalState.generator, p)
        return self

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
        return _FloatTensor()
#        return _FloatTensor_fromNative(newTensorC, False)

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

    def resize1d(_FloatTensor self, int size0):
        THFloatTensor_resize1d(self.thFloatTensor, size0)
        return self

    def resize2d(_FloatTensor self, int size0, int size1):
        THFloatTensor_resize2d(self.thFloatTensor, size0, size1)
        return self

    def resize3d(_FloatTensor self, int size0, int size1, int size2):
        THFloatTensor_resize3d(self.thFloatTensor, size0, size1, size2)
        return self

    def resize4d(_FloatTensor self, int size0, int size1, int size2, int size3):
        THFloatTensor_resize4d(self.thFloatTensor, size0, size1, size2, size3)
        return self

    def resize(_FloatTensor self, _FloatTensor size):
#        print('_FloatTensor.resize size:', size)
        if size.dims() == 0:
            return self
        cdef int dims = size.size()[0]
#        print('_FloatTensor.resize dims:', dims)
        if dims == 1:
            THFloatTensor_resize1d(self.thFloatTensor, size[0])
        elif dims == 2:
            THFloatTensor_resize2d(self.thFloatTensor, size[0], size[1])
        elif dims == 3:
            THFloatTensor_resize3d(self.thFloatTensor, size[0], size[1], size[2])
        elif dims == 4:
            THFloatTensor_resize4d(self.thFloatTensor, size[0], size[1], size[2], size[3])
        else:
            raise Exception('Not implemented for dims=' + str(dims))
        return self

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




cdef class _LongTensor(object):
    # properties are in the PyTorch.pxd file

#    def __cinit__(Tensor self, THFloatTensor *tensorC = NULL):
#        self.thFloatTensor = tensorC

    def __cinit__(self, *args, _allocate=True):
        print('LongTensor.__cinit__')
#        cdef THLongStorage *storageC
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
                print('no args, calling THLongTensor_new()')
                self.thLongTensor = THLongTensor_new()
            elif len(args) == 1:
#                print('new tensor 1d length', args[0])
                self.thLongTensor = THLongTensor_newWithSize1d(args[0])
#                storageC = THFloatTensor_storage(self.thFloatTensor)
#                if storageC == NULL:
#                    print('storageC is NULL')
#                else:
#                    print('storageC not null')
#                    addr = <long>(storageC)
#                    print('storageaddr', hex(addr))
#                    print('storageC refcount', THFloatStorage_getRefCount(storageC))
            elif len(args) == 2:
                self.thLongTensor = THLongTensor_newWithSize2d(args[0], args[1])
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
        refCount = THLongTensor_getRefCount(self.thLongTensor)
        print('LongTensor.dealloc old refcount', refCount)
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
        THLongTensor_free(self.thLongTensor)

    def nElement(_LongTensor self):
        return THLongTensor_nElement(self.thLongTensor)

    @property
    def refCount(_LongTensor self):
        return THLongTensor_getRefCount(self.thLongTensor)

    def geometric(_LongTensor self, float p=0.5):
        THLongTensor_geometric(self.thLongTensor, globalState.generator, p)
        return self

    cpdef int dims(self):
        return THLongTensor_nDimension(self.thLongTensor)

    cpdef set1d(self, int x0, long value):
        THLongTensor_set1d(self.thLongTensor, x0, value)

    cpdef set2d(self, int x0, int x1, long value):
        THLongTensor_set2d(self.thLongTensor, x0, x1, value)

    cpdef long get1d(self, int x0):
        return THLongTensor_get1d(self.thLongTensor, x0)

    cpdef long get2d(self, int x0, int x1):
        return THLongTensor_get2d(self.thLongTensor, x0, x1)





#class FloatTensor(_FloatTensor):
#    pass

#    @staticmethod
cdef _FloatTensor_fromNative(THFloatTensor *tensorC, retain=True):
    if retain:
        THFloatTensor_retain(tensorC)
    tensor = _FloatTensor(_allocate=False)
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

def _popFloatTensor():
    global globalState
    cdef THFloatTensor *tensorC = popFloatTensor(globalState.L)
    return _FloatTensor_fromNative(tensorC)

def _pushFloatTensor(_FloatTensor tensor):
    global globalState
    pushFloatTensor(globalState.L, tensor.thFloatTensor)

# there's probably an official Torch way of doing this
cpdef int getPrediction(_FloatTensor output):
    cdef int prediction = 0
    cdef float maxSoFar = output[0]
    cdef float thisValue = 0
    cdef int i = 0
    for i in range(THFloatTensor_size(output.thFloatTensor, 0)):
        thisValue = THFloatTensor_get1d(output.thFloatTensor, i)
        if thisValue > maxSoFar:
            maxSoFar = thisValue
            prediction = i
    return prediction + 1

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
