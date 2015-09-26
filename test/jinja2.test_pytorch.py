# {{header1}}
# {{header2}}

from __future__ import print_function
import PyTorch
import array
import numpy
import inspect

{% set types = {
    'Long': {'real': 'long'},
    'Float': {'real': 'float'}, 
    'Double': {'real': 'double'},
    'Byte': {'real': 'unsigned char'}
}
%}

def myeval(expr):
    parent_vars = inspect.stack()[1][0].f_locals
    print(expr, ':', eval(expr, parent_vars))

def myexec(expr):
    parent_vars = inspect.stack()[1][0].f_locals
    print(expr)
    exec(expr, parent_vars)

{% for Real in types %}
{% set real = types[Real]['real'] %}
def test_pytorch{{Real}}():
    PyTorch.manualSeed(123)
    numpy.random.seed(123)

    {{Real}}Tensor = PyTorch.{{Real}}Tensor

    {% if Real == 'Float' %}
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
    {% endif %}

    D = PyTorch.{{Real}}Tensor(5,3).fill(1)
    print('D', D)

    D[2][2] = 4
    print('D', D)

    D[3].fill(9)
    print('D', D)

    D.narrow(1,2,1).fill(0)
    print('D', D)

    {% if Real in ['Float', 'Double'] %}
    print(PyTorch.{{Real}}Tensor(3,4).uniform())
    print(PyTorch.{{Real}}Tensor(3,4).normal())
    print(PyTorch.{{Real}}Tensor(3,4).cauchy())
    print(PyTorch.{{Real}}Tensor(3,4).exponential())
    print(PyTorch.{{Real}}Tensor(3,4).logNormal())
    {% endif %}
    print(PyTorch.{{Real}}Tensor(3,4).bernoulli())
    print(PyTorch.{{Real}}Tensor(3,4).geometric())
    print(PyTorch.{{Real}}Tensor(3,4).geometric())
    PyTorch.manualSeed(3)
    print(PyTorch.{{Real}}Tensor(3,4).geometric())
    PyTorch.manualSeed(3)
    print(PyTorch.{{Real}}Tensor(3,4).geometric())

    print(type(PyTorch.{{Real}}Tensor(2,3)))

    size = PyTorch.LongTensor(2)
    size[0] = 4
    size[1] = 3
    D.resize(size)
    print('D after resize:\n', D)

    print('resize1d', PyTorch.{{Real}}Tensor().resize1d(3).fill(1))
    print('resize2d', PyTorch.{{Real}}Tensor().resize2d(2, 3).fill(1))
    print('resize', PyTorch.{{Real}}Tensor().resize(size).fill(1))

#    def myeval(expr):
#        print(expr, ':', eval(expr))

#    def myexec(expr):
#        print(expr)
#        exec(expr)

    myeval('{{Real}}Tensor(3,2).nElement()')
    myeval('{{Real}}Tensor().nElement()')
    myeval('{{Real}}Tensor(1).nElement()')

    A = {{Real}}Tensor(3,4).geometric(0.9)
    myeval('A')
    myexec('A += 3')
    myeval('A')
    myexec('A *= 3')
    myeval('A')
    {% if Real != 'Byte' %}
    myexec('A -= 3')
    {% endif %}
    myeval('A')
    myexec('A /= 3')
    myeval('A')

    myeval('A + 5')
    {% if Real != 'Byte' %}
    myeval('A - 5')
    {% endif %}
    myeval('A * 5')
    myeval('A / 2')

    B = {{Real}}Tensor().resizeAs(A).geometric(0.9)
    myeval('B')
    myeval('A + B')
    {% if Real != 'Byte' %}
    myeval('A - B')
    {% endif %}
    myexec('A += B')
    myeval('A')
    {% if Real != 'Byte' %}
    myexec('A -= B')
    myeval('A')
    {% endif %}

def test_pytorch_{{Real}}_constructors():
    {{Real}}Tensor = PyTorch.{{Real}}Tensor
    a = {{Real}}Tensor(3,2,5)
    assert(a.size().size() == 3)
    a = {{Real}}Tensor(3,2,5,6)
    assert(a.size().size() == 4)

def test_Pytorch_{{Real}}_operator_plus():
    {{Real}}Tensor = PyTorch.{{Real}}Tensor
    a = {{Real}}Tensor(3,2,5)
    b = {{Real}}Tensor(3,2,5)
    {%if Real in ['Float', 'Double'] %}
    a.uniform()
    b.uniform()
    {% else %}
    a.geometric(0.9)
    b.geometric(0.9)
    {% endif %}
    res = a + b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] + b.storage()[i])) < 0.000001)

def test_Pytorch_{{Real}}_operator_plusequals():
    {{Real}}Tensor = PyTorch.{{Real}}Tensor
    a = {{Real}}Tensor(3,2,5)
    b = {{Real}}Tensor(3,2,5)
    {%if Real in ['Float', 'Double'] %}
    a.uniform()
    b.uniform()
    {% else %}
    a.geometric(0.9)
    b.geometric(0.9)
    {% endif %}
    res = a.clone()
    res += b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] + b.storage()[i])) < 0.000001)

{% if Real not in ['Byte'] %}
def test_Pytorch_{{Real}}_operator_minus():
    {{Real}}Tensor = PyTorch.{{Real}}Tensor
    a = {{Real}}Tensor(3,2,5)
    b = {{Real}}Tensor(3,2,5)
    {%if Real in ['Float', 'Double'] %}
    a.uniform()
    b.uniform()
    {% else %}
    a.geometric(0.9)
    b.geometric(0.9)
    {% endif %}
    res = a - b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] - b.storage()[i])) < 0.000001)
{% endif %}

{% if Real not in ['Byte'] %}
def test_Pytorch_{{Real}}_operator_minusequals():
    {{Real}}Tensor = PyTorch.{{Real}}Tensor
    a = {{Real}}Tensor(3,2,5)
    b = {{Real}}Tensor(3,2,5)
    {%if Real in ['Float', 'Double'] %}
    a.uniform()
    b.uniform()
    {% else %}
    a.geometric(0.9)
    b.geometric(0.9)
    {% endif %}
    res = a.clone()
    res -= b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] - b.storage()[i])) < 0.000001)
{% endif %}

#def test_Pytorch_{{Real}}_cmul():
#    {{Real}}Tensor = PyTorch.{{Real}}Tensor
#    a = {{Real}}Tensor(3,2,5)
#    b = {{Real}}Tensor(3,2,5)
#    {%if Real in ['Float', 'Double'] %}
#    a.uniform()
#    b.uniform()
#    {% else %}
#    a.geometric(0.9)
#    b.geometric(0.9)
#    {% endif %}
#    res = a.cmul(b)
#    for i in range(3*2*5):
#        {% if Real == 'Byte' %}
#        assert(abs(res.storage()[i] - ((a.storage()[i] * b.storage()[i])) % 256) < 0.000001)
#        {% else %}
#        assert(abs(res.storage()[i] - (a.storage()[i] * b.storage()[i])) < 0.000001)
#        {% endif %}

def test_Pytorch_{{Real}}_cmul():
    {{Real}}Tensor = PyTorch.{{Real}}Tensor
    a = {{Real}}Tensor(3,2,5)
    b = {{Real}}Tensor(3,2,5)
    {%if Real in ['Float', 'Double'] %}
    a.uniform()
    b.uniform()
    {% else %}
    a.geometric(0.9)
    b.geometric(0.9)
    {% endif %}
    res = a.clone() #.cmul(b)
    res.cmul(b)
    for i in range(3*2*5):
        {% if Real == 'Byte' %}
        assert(abs(res.storage()[i] - ((a.storage()[i] * b.storage()[i])) % 256) < 0.000001)
        {% else %}
        assert(abs(res.storage()[i] - (a.storage()[i] * b.storage()[i])) < 0.000001)
        {% endif %}

def test_Pytorch_{{Real}}_operator_div():
    {{Real}}Tensor = PyTorch.{{Real}}Tensor
    a = {{Real}}Tensor(3,2,5)
    b = {{Real}}Tensor(3,2,5)
    {%if Real in ['Float', 'Double'] %}
    a.uniform()
    b.uniform()
    {% else %}
    a.geometric(0.9)
    b.geometric(0.9)
    {% endif %}
    res = a / b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] / b.storage()[i])) < 0.00001)

def test_Pytorch_{{Real}}_operator_divequals():
    {{Real}}Tensor = PyTorch.{{Real}}Tensor
    a = {{Real}}Tensor(3,2,5)
    b = {{Real}}Tensor(3,2,5)
    {%if Real in ['Float', 'Double'] %}
    a.uniform()
    b.uniform()
    {% else %}
    a.geometric(0.9)
    b.geometric(0.9)
    {% endif %}
    res = a.clone()
    res /= b
    for i in range(3*2*5):
        assert(abs(res.storage()[i] - (a.storage()[i] / b.storage()[i])) < 0.00001)


{% endfor %}

if __name__ == '__main__':
    {% for Real in types %}
    test_pytorch{{Real}}()
    {% endfor %}

