
all: simsearch.exe simsearch.so

simsearch.o: simsearch.cu
	nvcc $< -lcurand -o $@
	
simsearch.so: simsearch.cu
	nvcc $< -lcurand -o $@
