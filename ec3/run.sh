#!/usr/bin/bash
mkdir -p /tmp/png
# use the second 2080 Ti (tends to be the least loaded)
export CUDA_VISIBLE_DEVICES=2
_build/default/program.exe -b 768
# cat logo_log.txt
# ristretto test.png
# cd png 
# montage *.png -geometry +32+32 montage.jpg
