#!/bin/bash

# - torch is expected to be already activated, ie run:
#    source ~/torch/install/bin/torch_activate.sh
#    ... or similar
# - torch is expected to be at $HOME/torch

export PYTHONPATH=.:src
if [[ x$RUNGDB != x ]]; then {
    LUA_CPATH=$HOME/torch/install/lib/lua/5.1/?.so LD_LIBRARY_PATH=cbuild:$HOME/torch/install/lib:$HOME/torch/install/lib/lua/5.1 rungdb.sh python test/test_pynn.py
} elif [[ x$STDBUF != x ]]; then {
    LUA_CPATH=$HOME/torch/install/lib/lua/5.1/?.so LD_LIBRARY_PATH=cbuild:$HOME/torch/install/lib:$HOME/torch/install/lib/lua/5.1 stdbuf --output=L python test/test_pynn.py | tee test_outputs/test_pynn_output.txt
} else {
    LUA_CPATH=$HOME/torch/install/lib/lua/5.1/?.so LD_LIBRARY_PATH=cbuild:$HOME/torch/install/lib:$HOME/torch/install/lib/lua/5.1 python test/test_pynn.py
} fi

