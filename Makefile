CC=mpicc # Intel compiler
CFLAGS= -g -Wall 


BIN = .
LIB =  -lm -lmpi -fopenmp -lpthread
INC = -I.
HDR = $(wildcard *.h)
SRC = $(wildcard *.c)
OBJ = $(SRC:.c=.o)


all: clean main

%.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@ $(INC) $(LIB)

main:	$(OBJ)
	$(CC) $(CFLAGS) -o $(BIN)/main $(OBJ) $(LIB)

clean:
	find . -name "*.o"   -exec rm {} \;
	find . -name "*.c%"  -exec rm {} \;
	find . -name "*.bck" -exec rm {} \;
	find . -name "*~"    -exec rm {} \;
	find . -name "\#*"   -exec rm {} \;
	rm -f $(OBJ) main


