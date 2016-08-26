#!/bin/bash

# - must have Cython installed
# - must have already run:
#     mkdir cbuild
#     (cd cbuild && cmake .. && make -j 4 )
# - torch is expected to be already activated, ie run:
#    source ~/torch/install/bin/torch_activate.sh
#    ... or similar
# - torch is expected to be at $HOME/torch

export TORCH_INSTALL=$(dirname $(dirname $(which luajit) 2>/dev/null) 2>/dev/null)

if [[ x${INCREMENTAL} == x ]]; then {
  rm -Rf build PyBuild.so dist *.egg-info cbuild ${TORCH_INSTALL}/lib/libPyTorch*
  pip uninstall -y PyTorch
} fi
# python setup.py build_ext -i || exit 1

mkdir -p cbuild
if [[ x${TORCH_INSTALL} == x ]]; then {
    echo
    echo Please run:
    echo
    echo '    source ~/torch/install/bin/torch-activate'
    echo
    echo ... then try again
    echo
    exit 1
} fi

if [[ $(uname -s) == 'Darwin' ]]; then { USE_LUAJIT=OFF; } fi
if [[ x${USE_LUAJIT} == x ]]; then { USE_LUAJIT=ON; } fi
if [[ x${CYTHON} != x ]]; then { python setup.py cython_only || exit 1; } fi
(cd cbuild; cmake .. -DCMAKE_BUILD_TYPE=Debug -DUSE_LUAJIT=${USE_LUAJIT} -DCMAKE_INSTALL_PREFIX=${TORCH_INSTALL} && make -j 4 install) || exit 1

if [[ x${VIRTUAL_ENV} != x ]]; then {
    # we are in a virtualenv
    python setup.py install || exit 1
} else {
    # not virtualenv
    python setup.py install --user || exit 1
} fi
