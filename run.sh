#!/bin/bash

# - torch is expected to be already activated, ie run:
#    source ~/torch/install/bin/torch_activate.sh
#    ... or similar
# - torch is expected to be at $HOME/torch

LD_LIBRARY_PATH=$HOME/torch/install/lib:cbuild python test_pytorch.py

