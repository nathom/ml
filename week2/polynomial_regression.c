#include <math.h>

// #define NUM_SAMPLES 1000
// #define NUM_FEATURES DEGREE

#include "cos_data.h"
#include "matrix.h"
#include "multivariate_linear_regression.h"

void init_polynomal_matrix(matrix x_train, const matrix x_linear);

int
main()
{
    // (1,1000)
    matrix x_linear = matrix_from_buf((double *)x, 1, sizeof(x) / sizeof(double));
    // (1,1000)
    matrix y_train = matrix_from_buf((double *)cos_x, 1, sizeof(cos_x) / sizeof(double));

    // construct matrix [x x^2 ... x^DEGREE] into x_train
    // (1000, 3)
    matrix x_train;
    ZEROS(x_train, NUM_SAMPLES, NUM_FEATURES);
    init_polynomal_matrix(x_train, x_linear);
    printf("x_train: (%d,%d)\n", x_train.rows, x_train.cols);

    // normalize training data
    z_score_normalize(x_train);

    // gradient descent
    matrix w;
    ZEROS(w, NUM_FEATURES, 1);
    double b = 0.0;

    const int num_iterations = 1e5;
    const double alpha = 2;
    gradient_descent_ip(x_train, y_train, w, &b, num_iterations, alpha);

    printf("Final w:\n");
    matrix_print(w);
    printf("Final b: %f\n", b);
}

void
init_polynomal_matrix(matrix x_train, const matrix x_linear)
{
    for (int i = 0; i < NUM_SAMPLES; i++) {
        double x = matrix_get(x_linear, 0, i);
        for (int j = 0; j < NUM_FEATURES; j++) {
            matrix_set(x_train, i, j, pow(x, j + 1.0));
        }
    }
}
