
all: simsearch.exe libsimsearch.so

simsearch.exe: main.cpp libsimsearch.so
	g++ main.cpp -L. -lsimsearch -o $@
	
libsimsearch.so: simsearch.cu
	nvcc $< -lcurand -o $@ --compiler-options '-fPIC' --shared

clean:
	rm -rf simsearch.exe libsimsearch.so

# suggestion:
# export LD_LIBRARY_PATH=`pwd`
