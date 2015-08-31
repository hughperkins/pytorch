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
mlp.add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()

learningRate = 0.001

#mnist = fetch_mldata("MNIST original")
mndata = MNIST('/norep/data/mnist')
imagesList, labelsB = mndata.load_training()
images = numpy.array(imagesList).astype(numpy.float32)
#print('imagesArray', images.shape)

#print(images[0].shape)

labelsf = array.array('f', labelsB.tolist())
imagesTensor = PyTorch.asTensor(images)
#imagesTensor = PyTorch.FloatTensor(20,784)
#labels = numpy.array(20,).astype(numpy.int32)
#labels = PyTorch.FloatTensor(20).fill(1)
#print('labels', labels)
#print(imagesTensor.siz)

labelsTensor = PyTorch.asTensor(labelsf)
labelsTensor += 1
N = int(imagesTensor.size()[0])
for epoch in range(10):
    numRight = 0
    for n in range(N):
        print('n', n)
        input = imagesTensor[n]
        label = labelsTensor[n]
        print('label', label)
        labelTensor = PyTorch.FloatTensor(1)
        labelTensor[0] = label
        output = mlp.forward(input)
        prediction = mlp.getPrediction(output)
        if prediction == label:
            numRight += 1
        criterion.forward(output, labelTensor)
        mlp.zeroGradParameters()
        gradOutput = criterion.backward(output, labelTensor)
        mlp.backward(input, gradOutput)
        mlp.updateParameters(learningRate)
    print('epoch ' + str(epoch) + ' accuracy: ' + str(numRight * 100.0 / N) + '%')

