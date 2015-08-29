# pytorch
POC for wrapping torch in python

Example of what is possible currently:

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

