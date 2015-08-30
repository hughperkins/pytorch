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
print('weight', linear.getWeight())

#dataset = nn.Dataset()

#criterion = nn.MSECriterion()
#trainer = nn.StochasticGradient(linear, criterion)

sys.path.append('thirdparty/python-mnist')
from mnist import MNIST

mlp = nn.Linear(784, 10)

#mnist = fetch_mldata("MNIST original")
mndata = MNIST('/norep/data/mnist')
#imagesList, labels = mndata.load_training()
#images = numpy.array(imagesList).astype(numpy.float32)
#print('imagesArray', images.shape)

#print(images[0].shape)

#imagesTensor = PyTorch.asTensor(images)
imagesTensor = PyTorch.FloatTensor(20,784)
#print(imagesTensor.siz)

for n in range(20):
    print('n', n)
    thisImage = imagesTensor[n]
    print('thisImage.size()', thisImage.size())
    print('thisImage.storage.size()', thisImage.storage().size())
    mlpstorage = mlp.getOutput().storage()
    print('got mlpstorage')
    print('mlp.getOutput().storage is None', mlpstorage is None)
    output = mlp.updateOutput(thisImage)
    print('output.size()', output.size())
    print('thisImage.size()', thisImage.size())

