# pytorch
POC for wrapping torch in python

# Examples

These are a bit old now actually.  I should update them...

Examples of what is possible currently:
* pytorch
* pynn

# pytorch

Run by doing:
```
./build.sh
./run.sh
```

Script:
```
from __future__ import print_function
import PyTorch
import array
import numpy

A = numpy.random.rand(6).reshape(3,2).astype(numpy.float32)
B = numpy.random.rand(8).reshape(2,4).astype(numpy.float32)

C = A.dot(B)
print('C', C)

print('calling .asTensor...')
tensorA = PyTorch.asTensor(A)
tensorB = PyTorch.asTensor(B)
print(' ... asTensor called')

print('tensorA', tensorA)

tensorA.set2d(1, 1, 56.4)
tensorA.set2d(2, 0, 76.5)
print('tensorA', tensorA)
print('A', A)

tensorA += 5
print('tensorA', tensorA)
print('A', A)

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
```

Output:

```
C [[ 0.07321505  0.21034354  0.35347611  0.30827984]
 [ 0.1136106   0.21236189  0.37396491  0.12081985]
 [ 0.1540852   0.34465638  0.59388119  0.34144905]]
calling .asTensor...
 ... asTensor called
tensorA 0.108945 0.397354
0.697115 0.0342486
0.683192 0.335684
[torch.FloatTensor of size 3x2]

tensorA 0.108945 0.397354
0.697115 56.4
76.5 0.335684
[torch.FloatTensor of size 3x2]

A [[  0.10894503   0.3973541 ]
 [  0.697115    56.40000153]
 [ 76.5          0.33568445]]
tensorA 5.10894 5.39735
5.69711 61.4
81.5 5.33568
[torch.FloatTensor of size 3x2]

A [[  5.10894489   5.39735413]
 [  5.69711494  61.40000153]
 [ 81.5          5.3356843 ]]
tensorA2 12.1089 12.3974
12.6971 68.4
88.5 12.3357
[torch.FloatTensor of size 3x2]

tensorA 5.10894 5.39735
5.69711 61.4
81.5 5.33568
[torch.FloatTensor of size 3x2]

tensorAB 1.56072 3.88211 6.61399 4.68479
9.57568 29.3573 49.0571 46.1098
13.4707 25.4291 44.7226 15.1082
[torch.FloatTensor of size 3x4]

A.dot(B) [[  1.56071877   3.88210678   6.61398697   4.68478727]
 [  9.57568169  29.35725784  49.05712891  46.10975647]
 [ 13.47066402  25.42912674  44.72264099  15.10820866]]
tensorA[2] 81.5 5.33568
[torch.FloatTensor of size 2]

D 1 1 1
1 1 1
1 1 1
1 1 1
1 1 1
[torch.FloatTensor of size 5x3]

D 1 1 1
1 1 1
1 1 4
1 1 1
1 1 1
[torch.FloatTensor of size 5x3]

D 1 1 1
1 1 1
1 1 4
9 9 9
1 1 1
[torch.FloatTensor of size 5x3]

D 1 1 0
1 1 0
1 1 0
9 9 0
1 1 0
[torch.FloatTensor of size 5x3]
```

# pynn

Run by doing:
```
./build.sh
./nn_run.sh
```

Test script:
```
from __future__ import print_function
import PyTorch
import array
import numpy
import sys

A = numpy.random.rand(6).reshape(2,3).astype(numpy.float32)

tensorA = PyTorch.asTensor(A)

nn = PyTorch.Nn()
linear = nn.Linear(3, 8)
output = linear.updateOutput(tensorA)
print('output', output)
print('weight', linear.weight)

sys.path.append('thirdparty/python-mnist')
from mnist import MNIST

mlp = nn.Sequential()
linear = nn.Linear(784, 10)
mlp.add(linear)
logSoftMax = nn.LogSoftMax()
mlp.add(logSoftMax)

criterion = nn.ClassNLLCriterion()

learningRate = 0.0001

mndata = MNIST('/norep/data/mnist')
imagesList, labelsB = mndata.load_training()
images = numpy.array(imagesList).astype(numpy.float32)

labelsf = array.array('f', labelsB.tolist())
imagesTensor = PyTorch.asTensor(images)

labelsTensor = PyTorch.asTensor(labelsf)
labelsTensor += 1

desiredN = 128
imagesTensor = imagesTensor.narrow(0, 0, desiredN)
labelsTensor = labelsTensor.narrow(0, 0, desiredN)
print('imagesTensor.size()', imagesTensor.size())
print('labelsTensor.size()', labelsTensor.size())
N = int(imagesTensor.size()[0])

for epoch in range(10):
    numRight = 0
    for n in range(N):
        input = imagesTensor[n]
        label = labelsTensor[n]
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
        nn.collectgarbage()
    print('epoch ' + str(epoch) + ' accuracy: ' + str(numRight * 100.0 / N) + '%')
```

Output:
```
loaded lua library
output -0.52377 -0.186086 0.599191 -1.12187 0.380739 -0.101581 -0.268618 -0.0452895
-0.0496943 0.245387 0.384435 -0.860393 -0.00198464 -0.424576 -0.191637 -0.358058
[torch.FloatTensor of size 2x8]

weight 0.19025 -0.403272 0.179528
0.0461387 -0.372686 0.405773
-0.380512 0.127952 0.258736
-0.54166 -0.512322 -0.0570851
-0.403268 0.267247 0.0802541
-0.137717 0.229792 -0.344295
-0.239939 -0.118795 0.326363
0.00371041 0.391577 0.272496
[torch.FloatTensor of size 8x3]

nnWrapper.cpp ClassNLLCriterion::_ClassNLLCriterion type nn.ClassNLLCriterion
imagesTensor.size() 128 784
[torch.FloatTensor of size 2]

labelsTensor.size() 128
[torch.FloatTensor of size 1]

epoch 0 accuracy: 47.65625%
epoch 1 accuracy: 77.34375%
epoch 2 accuracy: 90.625%
epoch 3 accuracy: 94.53125%
epoch 4 accuracy: 96.09375%
epoch 5 accuracy: 100.0%
epoch 6 accuracy: 100.0%
epoch 7 accuracy: 100.0%
epoch 8 accuracy: 100.0%
epoch 9 accuracy: 100.0%
```

