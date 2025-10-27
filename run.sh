#!/bin/sh
PARENDI_ROOT=$(pwd) $HOME/.local/opt/verilator_parendi_ipu/bin/verilator_bin_dbg --poplar -O3 --dump-tree-dot --debugi 999 --build --tiles 4 $1