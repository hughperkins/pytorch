#!/bin/bash

# - torch is expected to be already activated, ie run:
#    source ~/torch/install/bin/torch-activate

if [[ x$RUNGDB != x ]]; then {
    rungdb.sh python $(which py.test) test/test_pytorch.py $*
} elif [[ x$STDBUF != x ]]; then {
    stdbuf --output=L py.test -sv test/test_pytorch.py $* | grep --line-buffered -v 'seconds =============' | tee test_outputs/test_pytorch_output.txt
} else {
    py.test -sv test/test_pytorch.py $*
} fi

