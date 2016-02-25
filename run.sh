#!/bin/bash

# - torch is expected to be already activated, ie run:
#    source ~/torch/install/bin/torch_activate.sh
#    ... or similar
# - torch is expected to be at $HOME/torch

# export PYTHONPATH=.:src

source ~/torch/install/bin/torch-activate

if [[ x$RUNGDB == x ]]; then {
    stdbuf --output=L python test/test_pytorch.py | tee test_outputs/test_pytorch_output.txt
} else {
    rungdb.sh test/python test_pytorch.py
} fi

