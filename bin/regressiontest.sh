#!/bin/bash

# for maintainer usage only; unsupported

if [[ ! -d envs/env34 ]]; then {
    virtualenv -p python3.4 envs/env34
} fi

if [[ ! -d envs/env27 ]]; then {
    virtualenv -p python2.7 envs/env27
} fi

for env in 27 34; do {
    source envs/env${env}/bin/activate
    source ~/torch/install/bin/torch-activate
    pip install wheel
    pip install -U pip
    pip install -U setuptools
    pip install pytest
    pip install numpy
    pip install cython
    pip install jinja2
    pip install python-mnist
    INCREMENTAL=1 CYTHON=1 ./build.sh
    py.test -sv test --ignore=env || exit 1
} done

