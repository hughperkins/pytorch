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
```

Output:

```
C [[ 0.00688038  0.42749292  0.6029557   0.03411297]
 [ 0.01043839  0.61446595  0.53244698  0.03617163]
 [ 0.00248079  0.18844125  0.60207909  0.02797816]]
calling .asTensor...
process2
('dims', 2)
rows=3 cols=2
allocate storage
allocate tensor
process2
('dims', 2)
rows=2 cols=4
allocate storage
allocate tensor
 ... asTensor called
tensorA 0.369563817978 0.45637011528
0.613313555717 0.283689081669
0.0802843496203 0.575758874416
[torch.FloatTensor of size 3x2]

tensorA 0.369563817978 0.45637011528
0.613313555717 56.4000015259
76.5 0.575758874416
[torch.FloatTensor of size 3x2]

A [[  0.36956382   0.45637012]
 [  0.61331356  56.40000153]
 [ 76.5          0.57575887]]
iadd
tensorA 5.3695640564 5.4563703537
5.61331367493 61.4000015259
81.5 5.57575893402
[torch.FloatTensor of size 3x2]

A [[  5.36956406   5.45637035]
 [  5.61331367  61.40000153]
 [ 81.5          5.57575893]]
iadd
allocate tensor
tensorA2 12.3695640564 12.4563703537
12.6133136749 68.4000015259
88.5 12.575758934
[torch.FloatTensor of size 3x2]

tensorA 5.3695640564 5.4563703537
5.61331367493 61.4000015259
81.5 5.57575893402
[torch.FloatTensor of size 3x2]

allocate tensor
allocate tensor
free tensor
tensorAB 0.0975383892655 5.97574090958 7.59979248047 0.444962382317
0.217197299004 17.4152946472 62.9951896667 2.86860704422
1.32064330578 75.2119216919 39.0041885376 3.42048835754
[torch.FloatTensor of size 3x4]

A.dot(B) [[  0.09753839   5.97574091   7.59979248   0.44496238]
 [  0.2171973   17.41529465  62.99518967   2.86860704]
 [  1.32064331  75.21192169  39.00418854   3.42048836]]
free tensor
free tensor
free storage
free tensor
free tensor
free storage
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
(There's a slight segfault at exit, but training is working now, as you see :-) )

