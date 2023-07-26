#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef NDEBUG
// Production builds should set NDEBUG=1
#define NDEBUG false
#else
#define NDEBUG true
#endif

#ifndef DEBUG
#define DEBUG !NDEBUG
#endif

typedef struct matrix {
    double *buf;
    int rows;
    int cols;
} matrix;

#define ZEROS(m, rows, cols)           \
    double _buf[rows][cols] = {{0.0}}; \
    m = matrix_from_buf((double *)_buf, rows, cols)

matrix matrix_new(int rows, int cols);
matrix matrix_from_buf(double *buf, int rows, int cols);
void matrix_del(matrix m);
void matrix_dot(matrix out, const matrix m1, const matrix m2);

static inline double
matrix_get(matrix m, int row, int col)
{
    if (DEBUG)
        if (!(row >= 0 && col >= 0 && row < m.rows && col < m.cols)) {
            printf("matrix get: Index out of bounds (%d,%d)\n", row, col);
            exit(1);
        }

    return m.buf[row * m.cols + col];
}

static inline void
matrix_set(matrix m, int row, int col, double val)
{
    if (DEBUG)
        if (!(row >= 0 && col >= 0 && row < m.rows && col < m.cols)) {
            printf("matrix set: Index out of bounds (%d,%d)\n", row, col);
            exit(1);
        }
    m.buf[row * m.cols + col] = val;
}

void matrix_print(const matrix m);
void matrix_print_dims(const matrix m);

// matrix matrix_transpose(matrix m);
matrix matrix_get_row(matrix m, const int row);
void matrix_scalar_multiply(matrix out, const matrix m, const double x);
void matrix_scalar_multiply_ip(matrix m, double x);
void matrix_add_ip(matrix m1, const matrix m2);
void matrix_add_ip_T(matrix m1, const matrix m2);
