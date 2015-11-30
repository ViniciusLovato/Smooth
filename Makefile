CC=gcc
MPICC =mpicc
NVCC =nvcc
OPENCV_FLAGS=`pkg-config --cflags opencv`
OPENCV_LIBS=`pkg-config --libs opencv`
O_LIBS = `cat path` 
OPENMP=-fopenmp -lm
OPT=-o2
BIN=bin
SRC=src

all: sequential parallel cuda
	
parallel:
	$(MPICC) $(OPENCV_FLAGS) $(OPT) -o $(BIN)/parallelSmooth $(SRC)/parallelSmooth.c $(O_LIBS) $(OPENMP)

sequential:
	$(CC) $(OPENCV_FLAGS) $(OPT) -o $(BIN)/smooth $(SRC)/smooth.c $(O_LIBS)

cuda:
	$(NVCC) $(OPENCV_FLAGS) -o $(BIN)/smoothCUDA $(SRC)/smoothCUDA.cu $(O_LIBS)

clean:
	rm -rf $(BIN)/smooth
	rm -rf $(BIN)/parallelSmooth
