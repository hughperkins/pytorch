#!/bin/bash

rm -Rf build *.so
python setup.py build_ext -i

