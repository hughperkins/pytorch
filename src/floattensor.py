from __future__ import print_function
import PyTorch

class FloatTensor(PyTorch._FloatTensor):
    pass
#    def __cinit__(self):
#        print('floattensor.__cinit__')

#    def __init__(self, tensor, _allocate=True):
#        print('floattensor.__init__')
#        if isinstance(tensor, PyTorch._FloatTensor):
#            self.native = tensor.native
#        else:
#            raise Exception('unknown type ' + type(tensor))

class DoubleTensor(PyTorch._DoubleTensor):
    pass
#    def __init__(self, tensor, _allocate=True):
#        print('doubletensor.__init__')
#        if isinstance(tensor, PyTorch._DoubleTensor):
#            self.native = tensor.native
#        else:
#            raise Exception('unknown type ' + type(tensor))

class LongTensor(PyTorch._LongTensor):
    pass

class ByteTensor(PyTorch._ByteTensor):
    pass

#class Linear(PyTorch.CyLinear):
#    pass

class FloatStorage(PyTorch._FloatStorage):
    pass

class DoubleStorage(PyTorch._DoubleStorage):
    pass

class LongStorage(PyTorch._LongStorage):
    pass

class ByteStorage(PyTorch._ByteStorage):
    pass

def asDoubleTensor(myarray):
    return DoubleTensor(PyTorch._asDoubleTensor(myarray))

def asFloatTensor(myarray):
    f1 = PyTorch._asFloatTensor(myarray)
    print('type(f1)', type(f1))
    return FloatTensor(f1)

