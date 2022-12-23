#!/usr/bin/bash
mkdir -p /tmp/png
# use the second 2080 Ti (tends to be the least loaded)
export OCAMLPARAM='_,rounds=4,O3=1,inline=100,inline-max-unroll=5'
export CUDA_VISIBLE_DEVICES=2
while getopts b: flag
do
	case "${flag}" in
		b) batch_size=${OPTARG};;
	esac
done
_build/default/program.exe -b $batch_size
# cat logo_log.txt
# ristretto test.png
# cd png 
# montage *.png -geometry +32+32 montage.jpg
