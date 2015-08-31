from __future__ import print_function
import PyTorch
import array
import numpy
#from sklearn.datasets import fetch_mldata
import sys

A = numpy.random.rand(6).reshape(2,3).astype(numpy.float32)

tensorA = PyTorch.asTensor(A)

nn = PyTorch.Nn()
linear = nn.Linear(3, 8)
output = linear.updateOutput(tensorA)
print('output', output)
print('weight', linear.weight)

#dataset = nn.Dataset()

#criterion = nn.MSECriterion()
#trainer = nn.StochasticGradient(linear, criterion)

sys.path.append('thirdparty/python-mnist')
from mnist import MNIST

mlp = nn.Sequential()
mlp.add(nn.Linear(784, 10))

criterion = nn.ClassNLLCriterion()

learningRate = 0.001

#mnist = fetch_mldata("MNIST original")
mndata = MNIST('/norep/data/mnist')
#imagesList, labels = mndata.load_training()
#images = numpy.array(imagesList).astype(numpy.float32)
#print('imagesArray', images.shape)

#print(images[0].shape)

#imagesTensor = PyTorch.asTensor(images)
imagesTensor = PyTorch.FloatTensor(20,784)
#labels = numpy.array(20,).astype(numpy.int32)
labels = PyTorch.FloatTensor(20).fill(1)
print('labels', labels)
#print(imagesTensor.siz)

for n in range(20):
    print('n', n)
    input = imagesTensor[n]
    label = labels[n]
    labelTensor = PyTorch.FloatTensor(1)
    labelTensor[0] = label
    output = mlp.forward(input)
    print('*** got output', output.size())
    criterion.forward(output, labelTensor)
    print('*** forwarded to criterion')
    mlp.zeroGradParameters()
    print('*** zerod grad parameters')
    gradOutput = criterion.backward(output, labelTensor)
    print('*** got gradOutput from criterion')
    mlp.backward(input, gradOutput)
    print('*** called backward on mlp')
    mlp.updateParameters(learningRate)
    print('*** called updatePrameters')

