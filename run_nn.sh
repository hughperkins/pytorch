#!/bin/bash

# - torch is expected to be already activated, ie run:
#    source ~/torch/install/bin/torch-activate

if [[ x$RUNGDB != x ]]; then {
    rungdb.sh python test/test_pynn.py
} elif [[ x$STDBUF != x ]]; then {
    stdbuf --output=L python test/test_pynn.py | tee test_outputs/test_pynn_output.txt
} else {
    python test/test_pynn.py
} fi

