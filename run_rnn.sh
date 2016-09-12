#!/bin/bash

if [[ x$RUNGDB != x ]]; then {
    rungdb.sh test/python prototyping/test_rnn.py
} else {
    python prototyping/test_rnn.py
} fi

