#define NUM_SAMPLES 3
#define NUM_FEATURES 4

#include "matrix.h"
#include "multivariate_linear_regression.h"

int
main()
{
    double x_train_buf[NUM_SAMPLES][NUM_FEATURES] = {
        {2104, 5, 1, 45}, {1416, 3, 2, 40}, {852, 2, 1, 35}};
    double y_train_buf[NUM_SAMPLES] = {460, 232, 178};

    matrix x_train = matrix_from_buf((double *)x_train_buf, 3, 4);
    matrix y_train = matrix_from_buf((double *)y_train_buf, 1, 3);

    // optimal values, should result in 0 cost
    double optimal_b = 785.1811367994083;
    double optimal_w_buf[] = {0.39133535, 18.75376741, -53.36032453, -26.42131618};
    matrix optimal_w = matrix_from_buf(optimal_w_buf, NUM_FEATURES, 1);

    // starting values for training
    double b = 0.0;
    double w_buf[NUM_FEATURES] = {0.0, 0.0, 0.0, 0.0};
    matrix w = matrix_from_buf(w_buf, NUM_FEATURES, 1);

    printf("Cost at optimal w: %.10e\n", compute_cost(x_train, y_train, optimal_w, optimal_b));

    matrix dj_dw;
    ZEROS(dj_dw, 1, NUM_FEATURES);
    struct grad g = {.dj_dw = dj_dw, .dj_db = 0.0};
    compute_gradient(&g, x_train, y_train, optimal_w, optimal_b);
    printf("Grad at optimal w: %.2e\n", g.dj_db);
    matrix_print(g.dj_dw);

    double cost = compute_cost(x_train, y_train, w, b);
    printf("Initial Cost: %f\n", cost);

    const int num_iterations = 50;
    // step size
    const double alpha = 1.0;

    // normalize input data, comment out and see the difference!
    // Without this, an alpha < 5e-7 and ~1e7 iterations are required for
    // it to converge
    z_score_normalize(x_train);

    // run gradient descent algorithm
    gradient_descent_ip(x_train, y_train, w, &b, num_iterations, alpha);

    printf("Final cost: %f\n", compute_cost(x_train, y_train, w, b));
    printf("Final w:\n");
    matrix_print(w);
    printf("Final b: %f\n", b);
    return 0;
}
