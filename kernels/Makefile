OBJ=sample_kernel

all:
	nvcc $(OBJ).cu --gpu-architecture=compute_75

clean:
	rm -f $(OBJ).o

cleanall: clean
	rm -f $(OBJ)