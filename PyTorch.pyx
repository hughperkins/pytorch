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
    void THFloatTensor_retain(THFloatTensor *self)

cdef class Tensor(object):
    cdef THFloatTensor *thFloatTensor
    cdef Storage storage  # Note that storing this here isnt very good, since we dont know when the underlying storage has been torn out 
                          # from under us...  Simply don't provide access to undelrying Storage objcts? Call retain?
                          # maybe leave like this for now, and fix it next time we have a segfault...

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

cdef extern from "nnWrapper.h":
    cdef struct lua_State
    lua_State *luaInit()
    void luaClose(lua_State *L)
    cdef cppclass _Linear:
        _Linear(lua_State *L, int inputSize, int OutputSize)
        void updateOutput(THFloatTensor *input)
        THFloatTensor *getOutput()

cdef class Linear(object):
    cdef _Linear *linear

    def __cinit__(self, Nn nn, inputSize, outputSize):
        self.linear = new _Linear(nn.L, inputSize, outputSize)

    def __dealloc__(self):
        del self.linear

    def updateOutput(self, Tensor input):
        self.linear.updateOutput(input.thFloatTensor)

    def getOutput(self):
        cdef THFloatTensor *outputC = self.linear.getOutput()
        print("PyTorch.pyx Linear.getOutput got output from c/lua layer")
        output = Tensor()
        output.thFloatTensor = outputC
        THFloatTensor_retain(output.thFloatTensor)
        cdef THFloatStorage *storageC = THFloatTensor_storage(outputC)
        storage = Storage()
        THFloatStorage_retain(storage.thFloatStorage)
        storage.thFloatStorage = storageC
        output.storage = storage
        return output

# cdef extern from "lua.h":

cdef class Nn(object):  # basically holds the Lua state, but maybe easier to call it Nn than LuaState?
    cdef lua_State *L

    def __cinit__(self):
        self.L = luaInit()
    
    def __dealloc__(self):
        luaClose(self.L)

    def Linear(self, inputSize, outputSize):
        cdef Linear linear = Linear(self, inputSize, outputSize)
        return linear

