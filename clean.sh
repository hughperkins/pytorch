#!/bin/bash

# This does a very deep clean of a lot of things.  I dont remember how safe it is...

export TORCH_INSTALL=$(dirname $(dirname $(which luajit) 2>/dev/null) 2>/dev/null)

rm -Rf build PyBuild.so dist *.egg-info cbuild src/*.so src/PyTorch.egg-info ${TORCH_INSTALL}/lib/libPyTorch*
pip uninstall -y PyTorch
rm src/Storage.cpp src/Storage.pxd src/Storage.pyx src/PyTorch.pxd src/PyTorch.pyx src/PyTorch.cpp
rm src/nnWrapper.cpp src/nnWrapper.h src/lua.pxd src/lua.pyx
