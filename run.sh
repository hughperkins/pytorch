#!/bin/bash

# - torch is expected to be already activated, ie run:
#    source ~/torch/install/bin/torch-activate

if [[ x$RUNGDB != x ]]; then {
    rungdb.sh test/python test_pytorch.py
} elif [[ -x$STDBUF != x ]]; then {
    stdbuf --output=L python test/test_pytorch.py | tee test_outputs/test_pytorch_output.txt
} else {
    python test/test_pytorch.py
} fi

