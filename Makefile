CC=gcc
OPENCV_FLAGS=`pkg-config --cflags opencv`
OPENCV_LIBS=`pkg-config --libs opencv`
OPENMP=-fopenmp -lm
BIN=bin
SRC=src

all:
	$(CC) $(OPENCV_FLAGS) -o $(BIN)/smooth $(SRC)/smooth.c $(OPENCV_LIBS)
	$(CC) $(OPENCV_FLAGS) -o $(BIN)/parallelSmooth $(SRC)/parallelSmooth.c $(OPENCV_LIBS) $(OPENMP)

parallel:
	$(CC) $(OPENCV_FLAGS) -o $(BIN)/paralleSmooth $(SRC)/parallelSmooth.c $(OPENCV_LIBS) $(OPENMP)

clean:
	rm -rf $(BIN)/smooth
	rm -rf $(BIN)/parallelSmooth
