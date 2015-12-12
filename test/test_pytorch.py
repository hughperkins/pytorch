# GENERATED FILE, do not edit by hand
# Source: test/jinja2.test_pytorch.py

from __future__ import print_function
import PyTorch
import array
import numpy
import inspect



def myeval(expr):
    parent_vars = inspect.stack()[1][0].f_locals
    print(expr, ':', eval(expr, parent_vars))

def myexec(expr):
    parent_vars = inspect.stack()[1][0].f_locals
    print(expr)
    exec(expr, parent_vars)



def test_pytorchDouble():
    PyTorch.manualSeed(123)
    numpy.random.seed(123)

    DoubleTensor = PyTorch.DoubleTensor

    

    D = PyTorch.DoubleTensor(5,3).fill(1)
    print('D', D)

    D[2][2] = 4
    print('D', D)

    D[3].fill(9)
    print('D', D)

    D.narrow(1,2,1).fill(0)
    print('D', D)

    
    print(PyTorch.DoubleTensor(3,4).uniform())
    print(PyTorch.DoubleTensor(3,4).normal())
    print(PyTorch.DoubleTensor(3,4).cauchy())
    print(PyTorch.DoubleTensor(3,4).exponential())
    print(PyTorch.DoubleTensor(3,4).logNormal())
    
    print(PyTorch.DoubleTensor(3,4).bernoulli())
    print(PyTorch.DoubleTensor(3,4).geometric())
    print(PyTorch.DoubleTensor(3,4).geometric())
    PyTorch.manualSeed(3)
    print(PyTorch.DoubleTensor(3,4).geometric())
    PyTorch.manualSeed(3)
    print(PyTorch.DoubleTensor(3,4).geometric())

    print(type(PyTorch.DoubleTensor(2,3)))

    size = PyTorch.LongStorage(2)
    size[0] = 4
    size[1] = 3
    D.resize(size)
    print('D after resize:\n', D)

    print('resize1d', PyTorch.DoubleTensor().resize1d(3).fill(1))
    print('resize2d', PyTorch.DoubleTensor().resize2d(2, 3).fill(1))
    print('resize', PyTorch.DoubleTensor().resize(size).fill(1))

    D = PyTorch.DoubleTensor(size).geometric()

#    def myeval(expr):
#        print(expr, ':', eval(expr))

#    def myexec(expr):
#        print(expr)
#        exec(expr)

    myeval('DoubleTensor(3,2).nElement()')
    myeval('DoubleTensor().nElement()')
    myeval('DoubleTensor(1).nElement()')

    A = DoubleTensor(3,4).geometric(0.9)
    myeval('A')
    myexec('A += 3')
    myeval('A')
    myexec('A *= 3')
    myeval('A')
    
    myexec('A -= 3')
    
    myeval('A')
    myexec('A /= 3')
    myeval('A')

    myeval('A + 5')
    
    myeval('A - 5')
    
    myeval('A * 5')
    myeval('A / 2')

    B = DoubleTensor().resizeAs(A).geometric(0.9)
    myeval('B')
    myeval('A + B')
    
    myeval('A - B')
    
    myexec('A += B')
    myeval('A')
    
    myexec('A -= B')
    myeval('A')
    

def test_pytorch_Double_constructors():
    DoubleTensor = PyTorch.DoubleTensor
    a = DoubleTensor(3,2,5)
    assert(len(a.size()) == 3)
    a = DoubleTensor(3,2,5,6)
    assert(len(a.size()) == 4)

def test_Pytorch_Double_operator_plus():
    DoubleTensor = PyTorch.DoubleTensor
    a = DoubleTensor(3,2,5)
    b = DoubleTensor(3,2,5)
    
    a.uniform()
    b.uniform()
    
    res = a + b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] + b.storage()[i])) < 0.000001)

def test_Pytorch_Double_operator_plusequals():
    DoubleTensor = PyTorch.DoubleTensor
    a = DoubleTensor(3,2,5)
    b = DoubleTensor(3,2,5)
    
    a.uniform()
    b.uniform()
    
    res = a.clone()
    res += b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] + b.storage()[i])) < 0.000001)


def test_Pytorch_Double_operator_minus():
    DoubleTensor = PyTorch.DoubleTensor
    a = DoubleTensor(3,2,5)
    b = DoubleTensor(3,2,5)
    
    a.uniform()
    b.uniform()
    
    res = a - b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] - b.storage()[i])) < 0.000001)



def test_Pytorch_Double_operator_minusequals():
    DoubleTensor = PyTorch.DoubleTensor
    a = DoubleTensor(3,2,5)
    b = DoubleTensor(3,2,5)
    
    a.uniform()
    b.uniform()
    
    res = a.clone()
    res -= b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] - b.storage()[i])) < 0.000001)


#def test_Pytorch_Double_cmul():
#    DoubleTensor = PyTorch.DoubleTensor
#    a = DoubleTensor(3,2,5)
#    b = DoubleTensor(3,2,5)
#    
#    a.uniform()
#    b.uniform()
#    
#    res = a.cmul(b)
#    for i in range(3*2*5):
#        
#        assert(abs(res.storage()[i] - (a.storage()[i] * b.storage()[i])) < 0.000001)
#        

def test_Pytorch_Double_cmul():
    DoubleTensor = PyTorch.DoubleTensor
    a = DoubleTensor(3,2,5)
    b = DoubleTensor(3,2,5)
    
    a.uniform()
    b.uniform()
    
    res = a.clone() #.cmul(b)
    res.cmul(b)
    for i in range(3*2*5):
        
        assert(abs(res.storage()[i] - (a.storage()[i] * b.storage()[i])) < 0.000001)
        

def test_Pytorch_Double_operator_div():
    DoubleTensor = PyTorch.DoubleTensor
    a = DoubleTensor(3,2,5)
    b = DoubleTensor(3,2,5)
    
    a.uniform()
    b.uniform()
    
    res = a / b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] / b.storage()[i])) < 0.00001)

def test_Pytorch_Double_operator_divequals():
    DoubleTensor = PyTorch.DoubleTensor
    a = DoubleTensor(3,2,5)
    b = DoubleTensor(3,2,5)
    
    a.uniform()
    b.uniform()
    
    res = a.clone()
    res /= b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] / b.storage()[i])) < 0.00001)




def test_pytorchByte():
    PyTorch.manualSeed(123)
    numpy.random.seed(123)

    ByteTensor = PyTorch.ByteTensor

    

    D = PyTorch.ByteTensor(5,3).fill(1)
    print('D', D)

    D[2][2] = 4
    print('D', D)

    D[3].fill(9)
    print('D', D)

    D.narrow(1,2,1).fill(0)
    print('D', D)

    
    print(PyTorch.ByteTensor(3,4).bernoulli())
    print(PyTorch.ByteTensor(3,4).geometric())
    print(PyTorch.ByteTensor(3,4).geometric())
    PyTorch.manualSeed(3)
    print(PyTorch.ByteTensor(3,4).geometric())
    PyTorch.manualSeed(3)
    print(PyTorch.ByteTensor(3,4).geometric())

    print(type(PyTorch.ByteTensor(2,3)))

    size = PyTorch.LongStorage(2)
    size[0] = 4
    size[1] = 3
    D.resize(size)
    print('D after resize:\n', D)

    print('resize1d', PyTorch.ByteTensor().resize1d(3).fill(1))
    print('resize2d', PyTorch.ByteTensor().resize2d(2, 3).fill(1))
    print('resize', PyTorch.ByteTensor().resize(size).fill(1))

    D = PyTorch.ByteTensor(size).geometric()

#    def myeval(expr):
#        print(expr, ':', eval(expr))

#    def myexec(expr):
#        print(expr)
#        exec(expr)

    myeval('ByteTensor(3,2).nElement()')
    myeval('ByteTensor().nElement()')
    myeval('ByteTensor(1).nElement()')

    A = ByteTensor(3,4).geometric(0.9)
    myeval('A')
    myexec('A += 3')
    myeval('A')
    myexec('A *= 3')
    myeval('A')
    
    myeval('A')
    myexec('A /= 3')
    myeval('A')

    myeval('A + 5')
    
    myeval('A * 5')
    myeval('A / 2')

    B = ByteTensor().resizeAs(A).geometric(0.9)
    myeval('B')
    myeval('A + B')
    
    myexec('A += B')
    myeval('A')
    

def test_pytorch_Byte_constructors():
    ByteTensor = PyTorch.ByteTensor
    a = ByteTensor(3,2,5)
    assert(len(a.size()) == 3)
    a = ByteTensor(3,2,5,6)
    assert(len(a.size()) == 4)

def test_Pytorch_Byte_operator_plus():
    ByteTensor = PyTorch.ByteTensor
    a = ByteTensor(3,2,5)
    b = ByteTensor(3,2,5)
    
    a.geometric(0.9)
    b.geometric(0.9)
    
    res = a + b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] + b.storage()[i])) < 0.000001)

def test_Pytorch_Byte_operator_plusequals():
    ByteTensor = PyTorch.ByteTensor
    a = ByteTensor(3,2,5)
    b = ByteTensor(3,2,5)
    
    a.geometric(0.9)
    b.geometric(0.9)
    
    res = a.clone()
    res += b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] + b.storage()[i])) < 0.000001)





#def test_Pytorch_Byte_cmul():
#    ByteTensor = PyTorch.ByteTensor
#    a = ByteTensor(3,2,5)
#    b = ByteTensor(3,2,5)
#    
#    a.geometric(0.9)
#    b.geometric(0.9)
#    
#    res = a.cmul(b)
#    for i in range(3*2*5):
#        
#        assert(abs(res.storage()[i] - ((a.storage()[i] * b.storage()[i])) % 256) < 0.000001)
#        

def test_Pytorch_Byte_cmul():
    ByteTensor = PyTorch.ByteTensor
    a = ByteTensor(3,2,5)
    b = ByteTensor(3,2,5)
    
    a.geometric(0.9)
    b.geometric(0.9)
    
    res = a.clone() #.cmul(b)
    res.cmul(b)
    for i in range(3*2*5):
        
        assert(abs(res.storage()[i] - ((a.storage()[i] * b.storage()[i])) % 256) < 0.000001)
        

def test_Pytorch_Byte_operator_div():
    ByteTensor = PyTorch.ByteTensor
    a = ByteTensor(3,2,5)
    b = ByteTensor(3,2,5)
    
    a.geometric(0.9)
    b.geometric(0.9)
    
    res = a / b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] / b.storage()[i])) < 0.00001)

def test_Pytorch_Byte_operator_divequals():
    ByteTensor = PyTorch.ByteTensor
    a = ByteTensor(3,2,5)
    b = ByteTensor(3,2,5)
    
    a.geometric(0.9)
    b.geometric(0.9)
    
    res = a.clone()
    res /= b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] / b.storage()[i])) < 0.00001)




def test_pytorchFloat():
    PyTorch.manualSeed(123)
    numpy.random.seed(123)

    FloatTensor = PyTorch.FloatTensor

    
    A = numpy.random.rand(6).reshape(3,2).astype(numpy.float32)
    B = numpy.random.rand(8).reshape(2,4).astype(numpy.float32)

    C = A.dot(B)
    print('C', C)

    print('calling .asTensor...')
    tensorA = PyTorch.asFloatTensor(A)
    tensorB = PyTorch.asFloatTensor(B)
    print(' ... asTensor called')

    print('tensorA', tensorA)

    tensorA.set2d(1, 1, 56.4)
    tensorA.set2d(2, 0, 76.5)
    print('tensorA', tensorA)
    print('A', A)

    print('add 5 to tensorA')
    tensorA += 5
    print('tensorA', tensorA)
    print('A', A)

    print('add 7 to tensorA')
    tensorA2 = tensorA + 7
    print('tensorA2', tensorA2)
    print('tensorA', tensorA)

    tensorAB = tensorA * tensorB
    print('tensorAB', tensorAB)

    print('A.dot(B)', A.dot(B))

    print('tensorA[2]', tensorA[2])
    

    D = PyTorch.FloatTensor(5,3).fill(1)
    print('D', D)

    D[2][2] = 4
    print('D', D)

    D[3].fill(9)
    print('D', D)

    D.narrow(1,2,1).fill(0)
    print('D', D)

    
    print(PyTorch.FloatTensor(3,4).uniform())
    print(PyTorch.FloatTensor(3,4).normal())
    print(PyTorch.FloatTensor(3,4).cauchy())
    print(PyTorch.FloatTensor(3,4).exponential())
    print(PyTorch.FloatTensor(3,4).logNormal())
    
    print(PyTorch.FloatTensor(3,4).bernoulli())
    print(PyTorch.FloatTensor(3,4).geometric())
    print(PyTorch.FloatTensor(3,4).geometric())
    PyTorch.manualSeed(3)
    print(PyTorch.FloatTensor(3,4).geometric())
    PyTorch.manualSeed(3)
    print(PyTorch.FloatTensor(3,4).geometric())

    print(type(PyTorch.FloatTensor(2,3)))

    size = PyTorch.LongStorage(2)
    size[0] = 4
    size[1] = 3
    D.resize(size)
    print('D after resize:\n', D)

    print('resize1d', PyTorch.FloatTensor().resize1d(3).fill(1))
    print('resize2d', PyTorch.FloatTensor().resize2d(2, 3).fill(1))
    print('resize', PyTorch.FloatTensor().resize(size).fill(1))

    D = PyTorch.FloatTensor(size).geometric()

#    def myeval(expr):
#        print(expr, ':', eval(expr))

#    def myexec(expr):
#        print(expr)
#        exec(expr)

    myeval('FloatTensor(3,2).nElement()')
    myeval('FloatTensor().nElement()')
    myeval('FloatTensor(1).nElement()')

    A = FloatTensor(3,4).geometric(0.9)
    myeval('A')
    myexec('A += 3')
    myeval('A')
    myexec('A *= 3')
    myeval('A')
    
    myexec('A -= 3')
    
    myeval('A')
    myexec('A /= 3')
    myeval('A')

    myeval('A + 5')
    
    myeval('A - 5')
    
    myeval('A * 5')
    myeval('A / 2')

    B = FloatTensor().resizeAs(A).geometric(0.9)
    myeval('B')
    myeval('A + B')
    
    myeval('A - B')
    
    myexec('A += B')
    myeval('A')
    
    myexec('A -= B')
    myeval('A')
    

def test_pytorch_Float_constructors():
    FloatTensor = PyTorch.FloatTensor
    a = FloatTensor(3,2,5)
    assert(len(a.size()) == 3)
    a = FloatTensor(3,2,5,6)
    assert(len(a.size()) == 4)

def test_Pytorch_Float_operator_plus():
    FloatTensor = PyTorch.FloatTensor
    a = FloatTensor(3,2,5)
    b = FloatTensor(3,2,5)
    
    a.uniform()
    b.uniform()
    
    res = a + b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] + b.storage()[i])) < 0.000001)

def test_Pytorch_Float_operator_plusequals():
    FloatTensor = PyTorch.FloatTensor
    a = FloatTensor(3,2,5)
    b = FloatTensor(3,2,5)
    
    a.uniform()
    b.uniform()
    
    res = a.clone()
    res += b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] + b.storage()[i])) < 0.000001)


def test_Pytorch_Float_operator_minus():
    FloatTensor = PyTorch.FloatTensor
    a = FloatTensor(3,2,5)
    b = FloatTensor(3,2,5)
    
    a.uniform()
    b.uniform()
    
    res = a - b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] - b.storage()[i])) < 0.000001)



def test_Pytorch_Float_operator_minusequals():
    FloatTensor = PyTorch.FloatTensor
    a = FloatTensor(3,2,5)
    b = FloatTensor(3,2,5)
    
    a.uniform()
    b.uniform()
    
    res = a.clone()
    res -= b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] - b.storage()[i])) < 0.000001)


#def test_Pytorch_Float_cmul():
#    FloatTensor = PyTorch.FloatTensor
#    a = FloatTensor(3,2,5)
#    b = FloatTensor(3,2,5)
#    
#    a.uniform()
#    b.uniform()
#    
#    res = a.cmul(b)
#    for i in range(3*2*5):
#        
#        assert(abs(res.storage()[i] - (a.storage()[i] * b.storage()[i])) < 0.000001)
#        

def test_Pytorch_Float_cmul():
    FloatTensor = PyTorch.FloatTensor
    a = FloatTensor(3,2,5)
    b = FloatTensor(3,2,5)
    
    a.uniform()
    b.uniform()
    
    res = a.clone() #.cmul(b)
    res.cmul(b)
    for i in range(3*2*5):
        
        assert(abs(res.storage()[i] - (a.storage()[i] * b.storage()[i])) < 0.000001)
        

def test_Pytorch_Float_operator_div():
    FloatTensor = PyTorch.FloatTensor
    a = FloatTensor(3,2,5)
    b = FloatTensor(3,2,5)
    
    a.uniform()
    b.uniform()
    
    res = a / b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] / b.storage()[i])) < 0.00001)

def test_Pytorch_Float_operator_divequals():
    FloatTensor = PyTorch.FloatTensor
    a = FloatTensor(3,2,5)
    b = FloatTensor(3,2,5)
    
    a.uniform()
    b.uniform()
    
    res = a.clone()
    res /= b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] / b.storage()[i])) < 0.00001)




def test_pytorchLong():
    PyTorch.manualSeed(123)
    numpy.random.seed(123)

    LongTensor = PyTorch.LongTensor

    

    D = PyTorch.LongTensor(5,3).fill(1)
    print('D', D)

    D[2][2] = 4
    print('D', D)

    D[3].fill(9)
    print('D', D)

    D.narrow(1,2,1).fill(0)
    print('D', D)

    
    print(PyTorch.LongTensor(3,4).bernoulli())
    print(PyTorch.LongTensor(3,4).geometric())
    print(PyTorch.LongTensor(3,4).geometric())
    PyTorch.manualSeed(3)
    print(PyTorch.LongTensor(3,4).geometric())
    PyTorch.manualSeed(3)
    print(PyTorch.LongTensor(3,4).geometric())

    print(type(PyTorch.LongTensor(2,3)))

    size = PyTorch.LongStorage(2)
    size[0] = 4
    size[1] = 3
    D.resize(size)
    print('D after resize:\n', D)

    print('resize1d', PyTorch.LongTensor().resize1d(3).fill(1))
    print('resize2d', PyTorch.LongTensor().resize2d(2, 3).fill(1))
    print('resize', PyTorch.LongTensor().resize(size).fill(1))

    D = PyTorch.LongTensor(size).geometric()

#    def myeval(expr):
#        print(expr, ':', eval(expr))

#    def myexec(expr):
#        print(expr)
#        exec(expr)

    myeval('LongTensor(3,2).nElement()')
    myeval('LongTensor().nElement()')
    myeval('LongTensor(1).nElement()')

    A = LongTensor(3,4).geometric(0.9)
    myeval('A')
    myexec('A += 3')
    myeval('A')
    myexec('A *= 3')
    myeval('A')
    
    myexec('A -= 3')
    
    myeval('A')
    myexec('A /= 3')
    myeval('A')

    myeval('A + 5')
    
    myeval('A - 5')
    
    myeval('A * 5')
    myeval('A / 2')

    B = LongTensor().resizeAs(A).geometric(0.9)
    myeval('B')
    myeval('A + B')
    
    myeval('A - B')
    
    myexec('A += B')
    myeval('A')
    
    myexec('A -= B')
    myeval('A')
    

def test_pytorch_Long_constructors():
    LongTensor = PyTorch.LongTensor
    a = LongTensor(3,2,5)
    assert(len(a.size()) == 3)
    a = LongTensor(3,2,5,6)
    assert(len(a.size()) == 4)

def test_Pytorch_Long_operator_plus():
    LongTensor = PyTorch.LongTensor
    a = LongTensor(3,2,5)
    b = LongTensor(3,2,5)
    
    a.geometric(0.9)
    b.geometric(0.9)
    
    res = a + b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] + b.storage()[i])) < 0.000001)

def test_Pytorch_Long_operator_plusequals():
    LongTensor = PyTorch.LongTensor
    a = LongTensor(3,2,5)
    b = LongTensor(3,2,5)
    
    a.geometric(0.9)
    b.geometric(0.9)
    
    res = a.clone()
    res += b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] + b.storage()[i])) < 0.000001)


def test_Pytorch_Long_operator_minus():
    LongTensor = PyTorch.LongTensor
    a = LongTensor(3,2,5)
    b = LongTensor(3,2,5)
    
    a.geometric(0.9)
    b.geometric(0.9)
    
    res = a - b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] - b.storage()[i])) < 0.000001)



def test_Pytorch_Long_operator_minusequals():
    LongTensor = PyTorch.LongTensor
    a = LongTensor(3,2,5)
    b = LongTensor(3,2,5)
    
    a.geometric(0.9)
    b.geometric(0.9)
    
    res = a.clone()
    res -= b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] - b.storage()[i])) < 0.000001)


#def test_Pytorch_Long_cmul():
#    LongTensor = PyTorch.LongTensor
#    a = LongTensor(3,2,5)
#    b = LongTensor(3,2,5)
#    
#    a.geometric(0.9)
#    b.geometric(0.9)
#    
#    res = a.cmul(b)
#    for i in range(3*2*5):
#        
#        assert(abs(res.storage()[i] - (a.storage()[i] * b.storage()[i])) < 0.000001)
#        

def test_Pytorch_Long_cmul():
    LongTensor = PyTorch.LongTensor
    a = LongTensor(3,2,5)
    b = LongTensor(3,2,5)
    
    a.geometric(0.9)
    b.geometric(0.9)
    
    res = a.clone() #.cmul(b)
    res.cmul(b)
    for i in range(3*2*5):
        
        assert(abs(res.storage()[i] - (a.storage()[i] * b.storage()[i])) < 0.000001)
        

def test_Pytorch_Long_operator_div():
    LongTensor = PyTorch.LongTensor
    a = LongTensor(3,2,5)
    b = LongTensor(3,2,5)
    
    a.geometric(0.9)
    b.geometric(0.9)
    
    res = a / b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] / b.storage()[i])) < 0.00001)

def test_Pytorch_Long_operator_divequals():
    LongTensor = PyTorch.LongTensor
    a = LongTensor(3,2,5)
    b = LongTensor(3,2,5)
    
    a.geometric(0.9)
    b.geometric(0.9)
    
    res = a.clone()
    res /= b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] / b.storage()[i])) < 0.00001)




if __name__ == '__main__':
    
    test_pytorchDouble()
    
    test_pytorchByte()
    
    test_pytorchFloat()
    
    test_pytorchLong()
    
