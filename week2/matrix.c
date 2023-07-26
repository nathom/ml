#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>

// implement 2d matrix

matrix
matrix_new(int rows, int cols)
{
    double *buf = calloc(rows * cols, sizeof(double));
    if (buf == NULL) {
        printf("Fail with calloc\n");
        exit(1);
    }

    return (matrix){
        .buf = buf,
        .rows = rows,
        .cols = cols,
    };
}

matrix
matrix_from_buf(double *buf, int rows, int cols)
{
    return (matrix){
        .buf = buf,
        .rows = rows,
        .cols = cols,
    };
}

void
matrix_del(matrix m)
{
    free(m.buf);
}

// O(n^3)
matrix
matrix_dot(matrix m1, matrix m2)
{
    if (DEBUG) {
        if (m1.cols != m2.rows) {
            printf("Dimension error %d!=%d\n", m1.cols, m2.rows);
            exit(1);
        }
    }
    matrix ret = matrix_new(m1.rows, m2.cols);
    for (int row = 0; row < m1.rows; row++) {
        for (int col = 0; col < m2.cols; col++) {
            double sum = 0.0;
            for (int k = 0; k < m1.cols; k++) {
                double x1 = matrix_get(m1, row, k);
                double x2 = matrix_get(m2, k, col);
                sum += x1 * x2;
            }
            matrix_set(ret, row, col, sum);
        }
    }
    return ret;
}

void
matrix_print(matrix m)
{
    for (int i = 0; i < m.rows; i++) {
        printf("[ ");
        for (int j = 0; j < m.cols; j++) {
            printf("%.4e", matrix_get(m, i, j));
            printf(" ");
        }
        printf("]\n");
    }
    printf("\n");
}

inline double
matrix_get(matrix m, int row, int col)
{
    if (DEBUG) {
        if (!(row >= 0 && col >= 0 && row < m.rows && col < m.cols)) {
            printf("matrix get: Index out of bounds (%d,%d)\n", row, col);
            exit(1);
        }
    }
    return m.buf[row * m.cols + col];
}

inline void
matrix_set(matrix m, int row, int col, double val)
{
    if (DEBUG)
        if (!(row >= 0 && col >= 0 && row < m.rows && col < m.cols)) {
            printf("matrix set: Index out of bounds (%d,%d)\n", row, col);
            exit(1);
        }
    m.buf[row * m.cols + col] = val;
}

matrix
matrix_get_row(matrix m, int row)
{
    if (DEBUG)
        if (row < 0 || row >= m.rows) {
            printf("Row %d out of bounds\n", row);
            exit(1);
        }
    return (matrix){.buf = m.buf + row * m.cols, .rows = 1, .cols = m.cols};
}

matrix
matrix_scalar_multiply(matrix m, double x)
{
    matrix ret = matrix_new(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            matrix_set(ret, i, j, matrix_get(m, i, j) * x);
        }
    }
    return ret;
}

void
matrix_scalar_multiply_ip(matrix m, double x)
{
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            matrix_set(m, i, j, matrix_get(m, i, j) * x);
        }
    }
}

void
matrix_add_ip(matrix m1, matrix m2)
{
    if (DEBUG)
        if (m1.rows != m2.rows || m1.cols != m2.cols) {
            printf("Add dimension mismatched\n");
            exit(1);
        }

    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.cols; j++) {
            double x1 = matrix_get(m1, i, j);
            double x2 = matrix_get(m2, i, j);
            matrix_set(m1, i, j, x1 + x2);
        }
    }
}

void
matrix_add_ip_T(matrix m1, matrix m2)
{
    if (DEBUG)
        if (m1.rows != m2.cols || m1.cols != m2.rows) {
            printf("Transpose add dimension mismatched\n");
            exit(1);
        }

    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.cols; j++) {
            double x1 = matrix_get(m1, i, j);
            double x2 = matrix_get(m2, j, i);
            matrix_set(m1, i, j, x1 + x2);
        }
    }
}

void
matrix_print_dims(matrix m)
{
    printf("(%d,%d)\n", m.rows, m.cols);
}
