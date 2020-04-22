NVCC=nvcc

all: clean install

install:
	$(NVCC) -arch=sm_35 *.cu -O3 -Xcompiler -fopenmp -lcublas -lcblas -o test -w


clean:
	rm -f test
