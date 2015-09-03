import PyTorch

class FloatTensor(PyTorch._FloatTensor):
    pass

#class Linear(PyTorch.CyLinear):
#    pass

class Linear(object):
    def __init__(self):
        print('Linear.__init__')

    def __attr__(self):
        print('Linear.__attr__')

