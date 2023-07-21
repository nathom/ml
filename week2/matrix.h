#include <stdio.h>
#include <stdlib.h>

typedef struct matrix {
    double *buf;
    int rows;
    int cols;
} matrix;

typedef matrix matrix_T;

matrix matrix_new(int rows, int cols);
matrix matrix_from_buf(double *buf, int rows, int cols);
void matrix_del(matrix m);
matrix matrix_dot(matrix m1, matrix m2);

double matrix_get(matrix m, int row, int col);
void matrix_set(matrix m, int row, int col, double val);
void matrix_print(matrix m);
// matrix matrix_transpose(matrix m);
matrix matrix_get_row(matrix m, int row);
matrix matrix_scalar_multiply(matrix m, double x);
void matrix_scalar_multiply_ip(matrix m, double x);
void matrix_add_ip(matrix m1, matrix m2);
void matrix_add_ip_T(matrix m1, matrix m2);
