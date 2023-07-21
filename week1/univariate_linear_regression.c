#include <stdio.h>
#include <stdlib.h>

struct gd_result compute_gradient(double *x, double *y, double w, double b, int m);
double mean_squared_error(double *x, double *y, double w, double b, int m);

struct gd_result {
    double d_dw;
    double d_db;
};

int
main()
{
    double x[] = {1, 2};
    double y[] = {300, 500};
    int x_size = sizeof(x) / sizeof(double);
    int y_size = sizeof(y) / sizeof(double);

    const double alpha = 0.01;
    const int num_iters = 1 << 13;

    if (x_size != y_size) {
        printf("Error: x size != y size\n");
        return 1;
    }

    // given the input/output pairs, we have to find w,b for the model
    // f_wb(x) = wx + b
    // that minimizes the squared error
    double w = 0.0, b = 0.0;
    printf("Initial error: %f\n", mean_squared_error(x, y, w, b, x_size));
    for (int i = 0; i < num_iters; i++) {
        if (i % (1 << 8) == 0) {
            printf("Error at i=%d: %f\n", i, mean_squared_error(x, y, w, b, x_size));
        }

        struct gd_result r = compute_gradient(x, y, w, b, x_size);
        w -= alpha * r.d_dw;
        b -= alpha * r.d_db;
    }

    printf("Final values w=%f, b=%f\n", w, b);
    printf("Final error %f\n", mean_squared_error(x, y, w, b, x_size));
}

// w, b in f_wb(x) = wx + b
double
mean_squared_error(double *x, double *y, double w, double b, int m)
{
    double squared_error = 0.0;
    for (int i = 0; i < m; i++) {
        double y_pred = w * x[i] + b;
        double diff = y_pred - y[i];
        squared_error += diff * diff;
    }
    return squared_error / (2 * m);
}

struct gd_result
compute_gradient(double *x, double *y, double w, double b, int m)
{
    double d_dw = 0.0, d_db = 0.0;
    for (int i = 0; i < m; i++) {
        double y_pred = w * x[i] + b;
        d_dw += (y_pred - y[i]) * x[i];
        d_db += (y_pred - y[i]);
    }
    return (struct gd_result){d_dw / m, d_db / m};
}
