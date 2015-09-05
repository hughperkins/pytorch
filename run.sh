#!/bin/bash

# - torch is expected to be already activated, ie run:
#    source ~/torch/install/bin/torch_activate.sh
#    ... or similar
# - torch is expected to be at $HOME/torch

if [[ x$RUNGDB == x ]]; then {
    LD_LIBRARY_PATH=$HOME/torch/install/lib:$PWD/cbuild python test_pytorch.py | tee test_pytorch_output.txt
} else {
    LD_LIBRARY_PATH=$HOME/torch/install/lib:$PWD/cbuild rungdb.sh python test_pytorch.py
} fi

