# if you type 'make' without arguments, this is the default
PROG = mlr
all: $(PROG)

# associated object files 
HEAD    = matrix.h
OBJ     = multivariate_linear_regression.o matrix.o

# special libraries
LIB	=
LIBFLAGS =

# select the compiler and flags you can over-ride on command line e.g. make DEBUG= 
CC      = gcc
DEBUG	= -ggdb
CSTD	=
WARN	= -Wall -Wextra -Werror
CDEFS	=
CFLAGS	= -I. $(DEBUG) $(WARN) $(CSTD) $(CDEFS)
AR      = ar
ARFLAGS = -rvs 

# some flags to use with assembler tools
ASLIST  = -Wa,-adhln
ASFLAGS =

# how to get an object dump of the target
DUMP   = objdump
DFLAGS = -d

$(OBJ):             $(HEAD)

# specify how to assemble a .S file
%.o : %.S
	$(CC) $(CFLAGS) $(ASFLAGS)  -c -o $@ $<

# specify how to compile the target
$(PROG):    $(OBJ) $(LIB)
	$(CC) $(CFLAGS) $(OBJ) $(LIBFLAGS) -o $@

.PHONY: clean

# get an object dump
dump:       $(PROG)
	$(DUMP) $(DFLAGS) $^

# remove binaries
.PHONY: clean
clean:
	rm -f $(OBJ) $(PROG)
