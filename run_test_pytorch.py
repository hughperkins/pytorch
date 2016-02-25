#!/bin/bash

# - torch is expected to be already activated, ie run:
#    source ~/torch/install/bin/torch_activate.sh
#    ... or similar
# - torch is expected to be at $HOME/torch

# export PYTHONPATH=.:src

source ~/torch/install/bin/torch-activate

if [[ x$RUNGDB == x ]]; then {
    stdbuf --output=L py.test -sv test/test_pytorch.py $* | grep --line-buffered -v 'seconds =============' | tee test_outputs/test_pytorch_output.txt
} else {
    rungdb.sh python $(which py.test) test/test_pytorch.py $*
} fi

