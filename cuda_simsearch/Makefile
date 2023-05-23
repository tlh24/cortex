
all: simsearch.exe libsimsearch.so simsearch2.exe

libsimsearch.so: simsearch.cu simsearch.h
	nvcc $< -lcurand -o $@ --compiler-options '-fPIC' --shared

simsearch.exe: main.cpp libsimsearch.so
	g++ main.cpp -L. -lsimsearch -o $@

# test both the C++ and C interfaces 
# (ocaml only works with C; Cuda is C++)
simsearch2.exe: main.c libsimsearch.so
	gcc main.c -L. -lsimsearch -o $@

clean:
	rm -rf simsearch.exe libsimsearch.so

# suggestion:
# export LD_LIBRARY_PATH=`pwd`
