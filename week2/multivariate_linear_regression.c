#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
struct grad {
    matrix dj_dw;
    double dj_db;
};

double compute_cost(matrix x_train, matrix y, matrix w, double b);
double predict(matrix x, matrix w, double b);
struct grad compute_gradient(matrix x_train, matrix y, matrix w, double b);
void gradient_descent_ip(matrix x_train, matrix y_train, matrix w, double *b, int num_iterations,
                         double alpha);

int
main()
{
    const int num_input_vars = 4;
    const int num_samples = 3;
    const int num_iterations = 1 << 14;
    const double alpha = 0.01;

    double x_train_buf[num_samples][num_input_vars] = {
        {2104, 5, 1, 45}, {1416, 3, 2, 40}, {852, 2, 1, 35}};
    double y_train_buf[num_samples] = {460, 232, 178};

    matrix x_train = matrix_from_buf((double *)x_train_buf, 3, 4);
    matrix y_train = matrix_from_buf((double *)y_train_buf, 1, 3);

    // optimal values, should result in 0 cost
    // double b = 785.1811367994083;
    // double w_buf[] = {0.39133535, 18.75376741, -53.36032453, -26.42131618};
    double b = 0.0;
    double w_buf[] = {0, 0, 0, 0};
    // column vector
    matrix w = matrix_from_buf(w_buf, num_input_vars, 1);

    double cost = compute_cost(x_train, y_train, w, b);
    printf("Initial Cost: %f\n", cost);

    gradient_descent_ip(x_train, y_train, w, &b, num_iterations, alpha);

    printf("Final cost: %f\n", compute_cost(x_train, y_train, w, b));
}

void
gradient_descent_ip(matrix x_train, matrix y_train, matrix w, double *b, int num_iterations,
                    double alpha)
{
    for (int i = 0; i < num_iterations; i++) {
        if (i % 100 == 0) {
            printf("Cost at %d: %f\n", i, compute_cost(x_train, y_train, w, *b));
        }
        struct grad g = compute_gradient(x_train, y_train, w, *b);
        matrix_scalar_multiply_ip(g.dj_dw, -alpha);
        // TODO: fix dimension mis match
        matrix_add_ip_T(w, g.dj_dw);
        *b -= alpha * g.dj_db;
        matrix_del(g.dj_dw);
    }
}

double
predict(matrix x, matrix w, double b)
{
    matrix dot_product = matrix_dot(x, w);
    double ret = dot_product.buf[0] + b;
    matrix_del(dot_product);
    return ret;
}

double
compute_cost(matrix x_train, matrix y, matrix w, double b)
{
    double cost = 0.0;
    for (int i = 0; i < x_train.rows; i++) {
        matrix dot = matrix_dot(matrix_get_row(x_train, i), w);
        double f_wb_mat = dot.buf[0] + b;
        double diff = y.buf[i] - f_wb_mat;
        cost += diff * diff;
        matrix_del(dot);
    }
    return cost / (2 * x_train.rows);
}

struct grad
compute_gradient(matrix x_train, matrix y, matrix w, double b)
{
    double dj_db = 0.0;
    // make it column for compatibility with w vector
    matrix dj_dw = matrix_new(1, x_train.cols);  // col vector of derivatives
    for (int i = 0; i < x_train.rows; i++) {
        matrix curr_row = matrix_get_row(x_train, i);
        matrix dot = matrix_dot(curr_row, w);

        double f_wb_mat = dot.buf[0] + b;
        double yi = matrix_get(y, 0, i);
        double err = f_wb_mat - yi;

        matrix scaled_row = matrix_scalar_multiply(curr_row, err);
        matrix_add_ip(dj_dw, scaled_row);
        matrix_del(scaled_row);

        dj_db += err;

        matrix_del(dot);
    }
    dj_db /= x_train.rows;
    matrix_scalar_multiply_ip(dj_dw, 1.0 / ((double)x_train.rows));

    // dJ/dw = \sum (f_wb - yi) \hat{x} (vector)
    // dJ/db = \sum (f_wb - yi)         (scalar)
    return (struct grad){.dj_dw = dj_dw, .dj_db = dj_db};
}
