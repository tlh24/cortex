#!/usr/bin/bash
mkdir -p /tmp/png
export OCAMLPARAM='_,rounds=4,O3=1,inline=100,inline-max-unroll=5'
# use the first 4090 (Second one for python)
export CUDA_VISIBLE_DEVICES=0
debug=false
parallel=false
while getopts b:gp flag
do
	case "${flag}" in
		b) batch_size=${OPTARG};;
		g) debug=true;;
		p) parallel=true;;
	esac
done
if "$debug"; then
	if "$parallel"; then 
		_build/default/program.exe -g -p -b $batch_size
	else 
		_build/default/program.exe -g -b $batch_size
	fi
else
	if "$parallel"; then 
		_build/default/program.exe -p -b $batch_size
	else 
		_build/default/program.exe -b $batch_size
	fi
fi
# cat logo_log.txt
# ristretto test.png
# cd png 
# montage *.png -geometry +32+32 montage.jpg
