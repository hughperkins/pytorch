import PyTorch

class FloatTensor(PyTorch._FloatTensor):
    pass

class DoubleTensor(PyTorch._DoubleTensor):
    pass

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

