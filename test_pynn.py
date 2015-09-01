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
linear = nn.Linear(784, 10)
mlp.add(linear)
logSoftMax = nn.LogSoftMax()
mlp.add(logSoftMax)

criterion = nn.ClassNLLCriterion()

learningRate = 0.001

#mnist = fetch_mldata("MNIST original")
mndata = MNIST('/norep/data/mnist')
#imagesList, labelsB = mndata.load_training()
#images = numpy.array(imagesList).astype(numpy.float32)
#print('imagesArray', images.shape)

#print(images[0].shape)

#labelsf = array.array('f', labelsB.tolist())
#imagesTensor = PyTorch.asTensor(images)

imagesTensor = PyTorch.FloatTensor(100,784)
#labels = numpy.array(20,).astype(numpy.int32)
labelsTensor = PyTorch.FloatTensor(100).fill(1)
#print('labels', labels)
#print(imagesTensor.siz)

def printStorageAddr(name, tensor):
    if tensor.storage() is None:
        print(name, 'storage is None')
    else:
        print(name, 'storage is ', hex(tensor.storage().dataAddr()))

#labelsTensor = PyTorch.asTensor(labelsf)
labelsTensor += 1
print('calling size on imagestensor...')
N = int(imagesTensor.size()[0])
print('   (called size)')
for epoch in range(10):
    numRight = 0
    for n in range(N):
        print('n', n)
        input = imagesTensor[n]
        label = labelsTensor[n]
        print('label', label)
        print('allocating 1 tensor')
        labelTensor = PyTorch.FloatTensor(1)
        print('(allocated)')
        print('labelTensor.refCount', labelTensor.refCount)
        printStorageAddr('** labelTensor', labelTensor)
        print('printingsize')
        print('labelTensor.size()', labelTensor.size())
        print('(printed) assigning label...')
        labelTensor[0] = label
        print('(assigned)')
        print('printstorageaddr...')
        printStorageAddr('labelTensor', labelTensor)
        print('(printed storageaddr)')
        print('logSoftMax.output.size().dims()', logSoftMax.output.size().dims())
        print('logSoftMax.output.size()', logSoftMax.output.size())
        printStorageAddr('logSoftMax.output', logSoftMax.output)
        output = mlp.forward(input)
        prediction = mlp.getPrediction(output)
        if prediction == label:
            numRight += 1
        print('output.size()', output.size())
        printStorageAddr('output', output)
        printStorageAddr('labelTensor', labelTensor)
        print('labelTensor.size()', labelTensor.size())
        print('labelTensor', labelTensor)
        criterion.forward(output, labelTensor)
        mlp.zeroGradParameters()
        gradOutput = criterion.backward(output, labelTensor)
        print('gradInput', criterion.gradInput.size())
        mlp.backward(input, gradOutput)
        mlp.updateParameters(learningRate)
    print('epoch ' + str(epoch) + ' accuracy: ' + str(numRight * 100.0 / N) + '%')

