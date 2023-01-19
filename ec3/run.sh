#!/usr/bin/bash
mkdir -p /tmp/png
export OCAMLPARAM='_,rounds=4,O3=1,inline=100,inline-max-unroll=5'
# use the first 4090 (Second one for python)
export CUDA_VISIBLE_DEVICES=0
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
