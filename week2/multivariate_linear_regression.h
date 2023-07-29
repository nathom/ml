#ifndef MULTIVARIATE_LINEAR_REGRESSION_H
#define MULTIVARIATE_LINEAR_REGRESSION_H

#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
struct grad {
    matrix dj_dw;
    double dj_db;
};

// #ifndef NUM_SAMPLES
// #error "Must define NUM_SAMPLES before include"
// #endif
// #ifndef NUM_FEATURES
// #error "Must define NUM_FEATURES before include"
// #endif

double compute_cost(matrix x_train, matrix y, matrix w, double b);
double predict(matrix x, matrix w, double b);
void compute_gradient(struct grad *g, const matrix x_train, const matrix y, const matrix w,
                      const double b);
void gradient_descent_ip(matrix x_train, matrix y_train, matrix w, double *b, int num_iterations,
                         double alpha);
void z_score_normalize(matrix x_train);
int mlr_demo();

#endif
