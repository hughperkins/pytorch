from __future__ import print_function
import PyTorch

from PyTorchAug import *

def test_pynn():
    PyTorch.manualSeed(123)
    linear = Linear(3, 5)
    linear
    print('linear', linear)
    print('linear.weight', linear.weight)
    print('linear.output', linear.output)
    print('linear.gradInput', linear.gradInput)

    input = PyTorch.DoubleTensor(4, 3).uniform()
    print('input', input)
    output = linear.updateOutput(input)
    print('output', output)

    gradInput = linear.updateGradInput(input, output)
    print('gradInput', gradInput)

    criterion = ClassNLLCriterion()
    print('criterion', criterion)

    print('dir(linear)', dir(linear))

    mlp = Sequential()
    mlp.add(linear)

    output = mlp.forward(input)
    print('output', output)

    import sys
    sys.path.append('thirdparty/python-mnist')
    from mnist import MNIST
    import numpy
    import array

    numpy.random.seed(123)

    mlp = Sequential()
    linear = Linear(784, 10)
    mlp.add(linear)
    logSoftMax = LogSoftMax()
    mlp.add(logSoftMax)
    mlp

    criterion = ClassNLLCriterion()
    print('got criterion')

    learningRate = 0.0001

    mndata = MNIST('/norep/data/mnist')
    imagesList, labelsB = mndata.load_training()
    images = numpy.array(imagesList).astype(numpy.float64)
    #print('imagesArray', images.shape)

    #print(images[0].shape)

    labelsf = array.array('d', labelsB.tolist())
    imagesTensor = PyTorch.asDoubleTensor(images)

    #imagesTensor = PyTorch.FloatTensor(100,784)
    #labels = numpy.array(20,).astype(numpy.int32)
    #labelsTensor = PyTorch.FloatTensor(100).fill(1)
    #print('labels', labels)
    #print(imagesTensor.size())

    def printStorageAddr(name, tensor):
        print('printStorageAddr START')
        storage = tensor.storage()
        if storage is None:
            print(name, 'storage is None')
        else:
            print(name, 'storage is ', hex(storage.dataAddr()))
        print('printStorageAddr END')

    labelsTensor = PyTorch.asDoubleTensor(labelsf)
    labelsTensor += 1
    #print('calling size on imagestensor...')
    #print('   (called size)')

    desiredN = 128
    maxN = int(imagesTensor.size()[0])
    desiredN = min(maxN, desiredN)
    imagesTensor = imagesTensor.narrow(0, 0, desiredN)
    labelsTensor = labelsTensor.narrow(0, 0, desiredN)
    print('imagesTensor.size()', imagesTensor.size())
    print('labelsTensor.size()', labelsTensor.size())
    N = int(imagesTensor.size()[0])
    print('type(imagesTensor)', type(imagesTensor))

    print('start training...')
    for epoch in range(4):
        numRight = 0
        for n in range(N):
    #        print('n', n)
            input = imagesTensor[n]
            label = labelsTensor[n]
            labelTensor = PyTorch.DoubleTensor(1)
            labelTensor[0] = label
    #        print('label', label)
            output = mlp.forward(input)
            prediction = PyTorch.getDoublePrediction(output)
    #        print('prediction', prediction)
            if prediction == label:
                numRight += 1
            criterion.forward(output, labelTensor)
            mlp.zeroGradParameters()
            gradOutput = criterion.backward(output, labelTensor)
            mlp.backward(input, gradOutput)
            mlp.updateParameters(learningRate)
    #        PyTorch.collectgarbage()
    #        if n % 100 == 0:
    #            print('n=', n)
        print('epoch ' + str(epoch) + ' accuracy: ' + str(numRight * 100.0 / N) + '%')

if __name__ == '__main__':
    test_pynn()

