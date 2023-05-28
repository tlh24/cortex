#!/usr/bin/bash
export LD_LIBRARY_PATH=`pwd`
export C_INCLUDE_PATH=`pwd` # needed for Ctypes static link build, fyi.
echo "verify that C++ and Ocaml all work equivalently."
./simsearch.exe
# ./simsearch2.exe
_build/default/simdb_test.exe
