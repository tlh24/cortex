#!/usr/bin/bash
# in another terminal, ./run.sh -b 512 -g etc. 
# use ps aux | grep main to get the pid & supply it as a command-line argument
# might be helfup to turn off O3 optimization in the compiler..
sudo perf record -F 99 -p $1 -g --call-graph=dwarf -- sleep 10
sudo chown tlh24 perf.data
perf script > perf.txt
/home/tlh24/var/FlameGraph/stackcollapse-perf.pl perf.txt > perf.folded
/home/tlh24/var/FlameGraph/flamegraph.pl perf.folded > perf.svg
