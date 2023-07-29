#include "multivariate_linear_regression.h"

#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"

void
gradient_descent_ip(const matrix x_train, const matrix y_train, matrix w, double *b,
                    const int num_iterations, const double alpha)
{
    // reusable buffer for gradient
    matrix dj_dw;
    ZEROS(dj_dw, 1, NUM_FEATURES);
    struct grad g = {.dj_dw = dj_dw, .dj_db = 0.0};

    for (int i = 0; i < num_iterations; i++) {
        if (i % (num_iterations >> 4) == 0) {
            printf("Cost at %d: %f\n", i, compute_cost(x_train, y_train, w, *b));
        }
        compute_gradient(&g, x_train, y_train, w, *b);
        // matrix_print(g.dj_dw);
        matrix_scalar_multiply_ip(g.dj_dw, -alpha);
        // matrix_print(g.dj_dw);
        // matrix_print(w);
        matrix_add_ip_T(w, g.dj_dw);
        // matrix_print(w);
        *b -= alpha * g.dj_db;
    }
}

double
predict(matrix x, matrix w, double b)
{
    matrix out;
    ZEROS(out, 1, 1);
    matrix_dot(out, x, w);
    double ret = out.buf[0] + b;
    return ret;
}

double
compute_cost(matrix x_train, matrix y, matrix w, double b)
{
    double cost = 0.0;
    for (int i = 0; i < x_train.rows; i++) {
        matrix x_i = matrix_get_row(x_train, i);
        double f_wb = predict(x_i, w, b);
        double diff = y.buf[i] - f_wb;
        cost += diff * diff;
    }
    return cost / (2.0 * x_train.rows);
}

void
compute_gradient(struct grad *g, const matrix x_train, const matrix y, const matrix w,
                 const double b)
{
    // dJ/dw = \sum (f_wb - yi) \hat{x} (vector)
    // dJ/db = \sum (f_wb - yi)         (scalar)
    matrix row_buf;
    ZEROS(row_buf, 1, NUM_FEATURES);
    for (int i = 0; i < x_train.rows; i++) {
        matrix curr_row = matrix_get_row(x_train, i);
        double f_wb = predict(curr_row, w, b);
        double yi = matrix_get(y, 0, i);
        double err = f_wb - yi;

        matrix_scalar_multiply(row_buf, curr_row, err);
        matrix_add_ip(g->dj_dw, row_buf);

        g->dj_db += err;
    }
    g->dj_db /= x_train.rows;
    matrix_scalar_multiply_ip(g->dj_dw, 1.0 / x_train.rows);
}

// normalize the input data using z-score
void
z_score_normalize(matrix x_train)
{
    // calculate mean and standard deviation
    matrix mean;
    ZEROS(mean, 1, NUM_FEATURES);
    matrix stdev;
    ZEROS(stdev, 1, NUM_FEATURES);

    for (int i = 0; i < NUM_SAMPLES; i++) {
        matrix_add_ip(mean, matrix_get_row(x_train, i));
    }
    matrix_scalar_multiply_ip(mean, 1.0 / NUM_SAMPLES);

    matrix buf;
    ZEROS(buf, 1, NUM_FEATURES);
    for (int i = 0; i < NUM_SAMPLES; i++) {
        matrix row = matrix_get_row(x_train, i);
        matrix_sub(buf, mean, row);
        matrix_square_ip(buf);
        matrix_add_ip(stdev, buf);
    }
    matrix_sqrt_ip(stdev);

    // normalize
    for (int i = 0; i < NUM_SAMPLES; i++) {
        matrix row = matrix_get_row(x_train, i);
        matrix_sub_ip(row, mean);
        matrix_div_ip(row, stdev);
    }
}
