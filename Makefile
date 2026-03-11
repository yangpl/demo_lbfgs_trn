CC ?= cc
CFLAGS ?= -O2 -g -Wall -Wextra -Wpedantic


BIN = .
LIB = -lm
INC = -I.
HDR = $(wildcard *.h)
SRC = $(wildcard *.c)
OBJ = $(SRC:.c=.o)


all: clean main

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ $(INC)

main:	$(OBJ)
	$(CC) $(CFLAGS) -o $(BIN)/main $(OBJ) $(LIB)

clean:
	find . -name "*.o"   -exec rm {} \;
	find . -name "*.c%"  -exec rm {} \;
	find . -name "*.bck" -exec rm {} \;
	find . -name "*~"    -exec rm {} \;
	find . -name "\#*"   -exec rm {} \;
	rm -f $(OBJ) main
