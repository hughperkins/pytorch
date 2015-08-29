#!/bin/bash

# must have Cython installed
# must have already run:
#   mkdir cbuild
#   (cd cbuild && cmake .. && make -j 4 )

rm -Rf build PyBuild.so
python setup.py build_ext -i

