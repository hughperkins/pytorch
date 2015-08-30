#!/bin/bash

# - torch is expected to be already activated, ie run:
#    source ~/torch/install/bin/torch_activate.sh
#    ... or similar
# - torch is expected to be at $HOME/torch

LUA_CPATH=$HOME/torch/install/lib/lua/5.1/?.so LD_LIBRARY_PATH=cbuild:$HOME/torch/install/lib:$HOME/torch/install/lib/lua/5.1 rungdb.sh python test_pynn.py

