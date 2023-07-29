# if you type 'make' without arguments, this is the default
PROG = mlr pr
all: $(PROG)

# associated object files 
HEAD    = matrix.h multivariate_linear_regression.h
OBJ     =  matrix.o multivariate_linear_regression.o
CFILES  = matrix.c multivariate_linear_regression.c

# special libraries
LIB	=
LIBFLAGS =

# select the compiler and flags you can over-ride on command line e.g. make DEBUG= 
CC      = cc
DEBUG	= -g
CSTD	=
WARN	= -Wall -Wextra -Werror
OPT     = -O3
CDEFS	=
CFLAGS	= -I. $(WARN) $(CSTD) $(CDEFS) $(OPT)
AR      = ar
ARFLAGS = -rvs 


# some flags to use with assembler tools
ASLIST  = -Wa,-adhln
ASFLAGS =

# how to get an object dump of the target
DUMP   = objdump
DFLAGS = -d

DEGREE = -DNUM_FEATURES=20

mlr: mlr.c $(CFILES) $(HEAD)
	$(CC) $(CFLAGS) mlr.c $(CFILES) -o $@ -DNUM_FEATURES=4 -DNUM_SAMPLES=3

pr: polynomial_regression.c $(CFILES) $(HEAD)
	$(CC) $(CFLAGS) polynomial_regression.c $(CFILES) -o $@ $(DEGREE) -DNUM_SAMPLES=1000

# specify how to assemble a .S file
%.o : %.S
	$(CC) $(CFLAGS) $(ASFLAGS)  -c -o $@ $<

# specify how to compile the target
# mlr: multivariate_linear_regression.o $(OBJ)
# 	$(CC) $(CFLAGS) $^ $(LIBFLAGS) -o $@ -DNUM_FEATURES=4 -DNUM_SAMPLES=3

clean:
	rm -f $(PROG) *.o
