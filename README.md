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
print('A', A)

tensorA += 5
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
C [[ 0.04153642  0.07916902  0.07554644  0.0697309 ]
 [ 0.16781303  0.37110865  0.28713757  0.27165675]
 [ 0.2460542   0.54121381  0.42204288  0.39888757]]
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
tensorA 0.16066198051 0.0498771443963
0.25907844305 0.364519119263
0.402094841003 0.52518415451
[torch.FloatTensor of size 3x2]

A [[  1.60661981e-01   4.98771444e-02]
 [  2.59078443e-01   5.64000015e+01]
 [  7.65000000e+01   5.25184155e-01]]
iadd
A [[  5.16066217   5.04987717]
 [  5.2590785   61.40000153]
 [ 81.5          5.52518415]]
iadd
allocate tensor
tensorA2 12.1606616974 12.0498771667
12.2590789795 68.4000015259
88.5 12.5251846313
[torch.FloatTensor of size 3x2]

tensorA 5.16066217422 5.04987716675
5.25907850266 61.4000015259
81.5 5.52518415451
[torch.FloatTensor of size 3x2]

allocate tensor
allocate tensor
free tensor
tensorAB 2.55792856216 5.49748468399 4.43292808533 4.17206001282
22.573091507 53.8077049255 37.2520561218 35.7777557373
14.051158905 23.2137203217 26.8149089813 24.2896614075
[torch.FloatTensor of size 3x4]

A.dot(B) [[  2.55792856   5.49748468   4.43292809   4.17206001]
 [ 22.57309151  53.80770493  37.25205612  35.77775574]
 [ 14.05115891  23.21372032  26.81490898  24.28966141]]
free tensor
free tensor
free storage
free tensor
free tensor
free storage
```

