
all: simsearch.exe simsearch.so

simsearch.exe: simsearch.cu
	nvcc $< -lcurand -o $@
	
simsearch.so: simsearch.cu
	nvcc $< -lcurand -o $@

clean:
	rm -rf simsearch.exe simsearch.so
