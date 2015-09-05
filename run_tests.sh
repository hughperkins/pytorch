#!/bin/bash

# - torch is expected to be already activated, ie run:
#    source ~/torch/install/bin/torch_activate.sh
#    ... or similar
# - torch is expected to be at $HOME/torch

export PYTHONPATH=.

if [[ x$RUNGDB == x ]]; then {
    LD_LIBRARY_PATH=$HOME/torch/install/lib:$PWD/cbuild py.test -sv test/test* | tee tests_output.txt
} else {
    LD_LIBRARY_PATH=$HOME/torch/install/lib:$PWD/cbuild rungdb.sh python $(which py.test) test/test*
} fi

