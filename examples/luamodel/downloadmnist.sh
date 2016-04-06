#!/bin/bash
#
# download mnist files

echo downloading mnist data...
pushd ../..
mkdir -p data/mnist
(cd data/mnist
  if [[ ! -f train-images-idx3-ubyte ]]; then {
    wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O train-images-idx3-ubyte.gz
    gunzip train-images-idx3-ubyte.gz
  } fi

  if [[ ! -f train-labels-idx1-ubyte ]]; then {
    wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O train-labels-idx1-ubyte.gz
    gunzip train-labels-idx1-ubyte.gz
  } fi

  if [[ ! -f t10k-images-idx3-ubyte ]]; then {
    wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O t10k-images-idx3-ubyte.gz
    gunzip t10k-images-idx3-ubyte.gz
  } fi

  if [[ ! -f t10k-labels-idx1-ubyte ]]; then {
    wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -O t10k-labels-idx1-ubyte.gz
    gunzip t10k-labels-idx1-ubyte.gz
  } fi
)  
 echo ...downloaded mnist data

