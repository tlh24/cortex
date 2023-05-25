#!/usr/bin/bash
# might be helfup to turn off O3 optimization in the compiler..
pid=$(pgrep -f main.exe | head -n 1)
echo "capturing 10 sec of data from process $pid"
sudo perf record -F 99 -p $pid -g --call-graph=dwarf -- sleep 10
sudo chown tlh24 perf.data
echo "processing the perf binary data"
perf script > perf.txt
/home/tlh24/var/FlameGraph/stackcollapse-perf.pl perf.txt > perf.folded
/home/tlh24/var/FlameGraph/flamegraph.pl perf.folded > perf.svg
echo "launching firefox window to view flame graph"
# Open the file in a new private Firefox window
firefox --private-window file:perf.svg
