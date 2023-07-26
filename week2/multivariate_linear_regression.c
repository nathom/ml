#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
struct grad {
    matrix dj_dw;
    double dj_db;
};

double compute_cost(matrix x_train, matrix y, matrix w, double b);
double predict(matrix x, matrix w, double b);
void compute_gradient(struct grad *g, const matrix x_train, const matrix y, const matrix w,
                      const double b);
void gradient_descent_ip(matrix x_train, matrix y_train, matrix w, double *b, int num_iterations,
                         double alpha);

#define NUM_INPUT_VARS 4
#define NUM_SAMPLES 3

int
main()
{
    double x_train_buf[NUM_SAMPLES][NUM_INPUT_VARS] = {
        {2104, 5, 1, 45}, {1416, 3, 2, 40}, {852, 2, 1, 35}};
    double y_train_buf[NUM_SAMPLES] = {460, 232, 178};

    matrix x_train = matrix_from_buf((double *)x_train_buf, 3, 4);
    matrix y_train = matrix_from_buf((double *)y_train_buf, 1, 3);

    // optimal values, should result in 0 cost
    double optimal_b = 785.1811367994083;
    double optimal_w_buf[] = {0.39133535, 18.75376741, -53.36032453, -26.42131618};
    matrix optimal_w = matrix_from_buf(optimal_w_buf, NUM_INPUT_VARS, 1);

    // starting values for training
    double b = 0.0;
    double w_buf[NUM_INPUT_VARS] = {0.0, 0.0, 0.0, 0.0};
    matrix w = matrix_from_buf(w_buf, NUM_INPUT_VARS, 1);

    printf("Sample prediction: %f\n", predict(matrix_get_row(x_train, 0), optimal_w, optimal_b));
    printf("Cost at optimal w: %.10e\n", compute_cost(x_train, y_train, optimal_w, optimal_b));
    matrix dj_dw;
    ZEROS(dj_dw, 1, NUM_INPUT_VARS);
    struct grad g = {.dj_dw = dj_dw, .dj_db = 0.0};
    compute_gradient(&g, x_train, y_train, optimal_w, optimal_b);
    printf("Grad at optimal w: %.2e\n", g.dj_db);
    matrix_print(g.dj_dw);

    double cost = compute_cost(x_train, y_train, w, b);
    printf("Initial Cost: %f\n", cost);

    const int num_iterations = 1e8;
    // step size
    const double alpha = 5e-7;

    // run gradient descent algorithm
    gradient_descent_ip(x_train, y_train, w, &b, num_iterations, alpha);

    printf("Final cost: %f\n", compute_cost(x_train, y_train, w, b));
    printf("Final w:\n");
    matrix_print(w);
    printf("Final b: %f\n", b);

    printf("Post training predictions:\n");
    for (int i = 0; i < NUM_SAMPLES; i++) {
        matrix row = matrix_get_row(x_train, i);
        printf("Got %f, expected %f\n", predict(row, w, b), predict(row, optimal_w, optimal_b));
    }
}

void
gradient_descent_ip(const matrix x_train, const matrix y_train, matrix w, double *b,
                    const int num_iterations, const double alpha)
{
    // reusable buffer for gradient
    matrix dj_dw;
    ZEROS(dj_dw, 1, NUM_INPUT_VARS);
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
    ZEROS(row_buf, 1, NUM_INPUT_VARS);
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
