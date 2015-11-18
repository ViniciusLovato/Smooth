CC=gcc
MPICC =mpicc
NVCC =nvcc
OPENCV_FLAGS=`pkg-config --cflags opencv`
OPENCV_LIBS=`pkg-config --libs opencv`
O_LIBS = `cat path` 
OPENMP=-fopenmp -lm
OPT=-o2 -g
BIN=bin
SRC=src

all:
	$(CC) $(OPENCV_FLAGS) $(OPT) -o $(BIN)/smooth $(SRC)/smooth.c $(O_LIBS)
	$(NVCC) $(OPENCV_FLAGS) -o $(BIN)/smoothCUDA $(SRC)/smoothCUDA.cu $(O_LIBS)
	$(MPICC) $(OPENCV_FLAGS) $(OPT) -o $(BIN)/parallelSmooth $(SRC)/parallelSmooth.c $(O_LIBS) $(OPENMP)

parallel:
	$(MPICC) $(OPENCV_FLAGS) -o $(BIN)/paralleSmooth $(SRC)/parallelSmooth.c $(OPENCV_LIBS) $(OPENMP)

clean:
	rm -rf $(BIN)/smooth
	rm -rf $(BIN)/parallelSmooth
