# {{header1}}
# {{header2}}

from __future__ import print_function
import PyTorch
import array
import numpy

{% set types = {
    'Long': {'real': 'long'},
    'Float': {'real': 'float'}, 
    'Double': {'real': 'double'},
    'Byte': {'real': 'unsigned char'}
}
%}

{% for Real in types %}
{% set real = types[Real]['real'] %}
def test_pytorch{{Real}}():
    PyTorch.manualSeed(123)
    numpy.random.seed(123)

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

    def myeval(expr):
        print(expr, ':', eval(expr))

    myeval('PyTorch.{{Real}}Tensor(3,2).nElement()')
    myeval('PyTorch.{{Real}}Tensor().nElement()')
    myeval('PyTorch.{{Real}}Tensor(1).nElement()')
{% endfor %}

if __name__ == '__main__':
    {% for Real in types %}
    test_pytorch{{Real}}()
    {% endfor %}

