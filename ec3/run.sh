#!/usr/bin/bash
rm -rf /tmp/ec3
mkdir -p /tmp/ec3
mkdir -p /tmp/ec3/render_simplest
mkdir -p /tmp/ec3/vae_samp
mkdir -p /tmp/ec3/replace_verify
mkdir -p /tmp/ec3/mnist_improve
mkdir -p /tmp/ec3/init_database
export OCAMLPARAM='_,rounds=4,O3=1,inline=100,inline-max-unroll=5'
# use the first 4090 (Second one for python)
export CUDA_VISIBLE_DEVICES=0
debug=false
parallel=false
timing=false
while getopts b:gp flag
do
	case "${flag}" in
		b) batch_size=${OPTARG};;
		g) debug=true;;
		p) parallel=true;;
		t) timing=true;;
	esac
done
if "$debug"; then
	if "$parallel"; then 
		if "$timing"; then 
			_build/default/program.exe -g -p -t -b $batch_size
		else 
			_build/default/program.exe -g -p -b $batch_size
		fi
	else 
		if "$timing"; then 
			_build/default/program.exe -g -t -b $batch_size
		else 
			_build/default/program.exe -g -b $batch_size
		fi
	fi
else
	if "$parallel"; then 
		if "$timing"; then 
			_build/default/program.exe -p -t -b $batch_size
		else 
			_build/default/program.exe -p -b $batch_size
		fi
	else 
		if "$timing"; then 
			_build/default/program.exe -t -b $batch_size
		else 
			_build/default/program.exe -b $batch_size
		fi
	fi
fi
# cat logo_log.txt
# ristretto test.png
# cd png 
# montage *.png -geometry +32+32 montage.jpg
