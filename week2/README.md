# Week 2

Multivariate linear regression, feature scaling, and polynomial
regression.

### Multivariate linear regression

Run the test binary `mlr` with

```
make mlr && ./mlr
```

This will run the algorithm on a small data set.

### Polynomial regression

Run the test binary `pr` with

```
make pr && ./pr
```

This will fit a 15th degree polynomial to cos(x). That data is stored
in `cos_data.h`.

## Putting your own data and parameters in

Because the files I'm using assume fixed size input, you need to change
the NUM_SAMPLES and NUM_FEATURES macro values in the makefile. Then plug in
your own data and run!
