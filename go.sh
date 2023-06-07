#!/usr/bin/bash
eval $(opam env)
source ~/venv311/bin/activate
cd cortex/ec3/
# export OCAMLPARAM='_,rounds=4,O3=1,inline=100,inline-max-unroll=5'
# too much optimization. slows the system down.
export OCAMLPARAM='_,O3=1' # ,inline=20 makes things easier to read, but slower
export OCAMLRUNPARAM=b #debugging
export LIBTORCH=/home/tlh24/var/libtorch/
