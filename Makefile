CC=gcc
OPENCV_FLAGS=`pkg-config --cflags opencv`
OPENCV_LIBS=`pkg-config --libs opencv`
BIN=bin
SRC=src

all:
	$(CC) $(OPENCV_FLAGS) -o $(BIN)/smooth $(SRC)/smooth.c $(OPENCV_LIBS)

clean:
	rm -rf $(BIN)/smooth
