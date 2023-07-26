/**
 * File: matrix.h
 * Description: This file implements basic operations on a 2D matrix, such as creation,
 *              deletion, dot product, scalar multiplication, and addition.
 */
#ifndef MATRIX_H
#define MATRIX_H

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

#define ZEROS(m, rows, cols)           \
    double _buf[rows][cols] = {{0.0}}; \
    m = matrix_from_buf((double *)_buf, rows, cols)

typedef struct {
    double *buf;
    int rows;
    int cols;
} matrix;

/**
 * Function: matrix_get
 * Description: Retrieves the value of a matrix element at the specified row and column.
 *
 * Parameters:
 *   m: The matrix from which to retrieve the element value.
 *   row: The row index of the element to retrieve.
 *   col: The column index of the element to retrieve.
 *
 * Returns:
 *   The value of the matrix element at the specified row and column.
 *
 * Notes:
 *   - This function assumes that the row and column indices are valid (non-negative
 *     and within the bounds of the matrix).
 *   - If DEBUG is enabled (non-zero), this function checks if the indices are within
 *     bounds. If an index is out of bounds, it prints an error message and exits the
 *     program. Otherwise, it directly accesses the matrix element and returns its value.
 */
static inline double
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

/**
 * Function: matrix_set
 * Description: Sets the value of a matrix element at the specified row and column.
 *
 * Parameters:
 *   m: The matrix in which to set the element value.
 *   row: The row index of the element to set.
 *   col: The column index of the element to set.
 *   val: The value to be assigned to the matrix element.
 *
 * Notes:
 *   - This function assumes that the row and column indices are valid (non-negative
 *     and within the bounds of the matrix).
 *   - If DEBUG is enabled (non-zero), this function checks if the indices are within
 *     bounds. If an index is out of bounds, it prints an error message and exits the
 *     program. Otherwise, it directly accesses the matrix element and sets its value
 *     to the specified 'val'.
 */
static inline void
matrix_set(matrix m, int row, int col, double val)
{
    if (DEBUG) {
        if (!(row >= 0 && col >= 0 && row < m.rows && col < m.cols)) {
            printf("matrix set: Index out of bounds (%d,%d)\n", row, col);
            exit(1);
        }
    }

    m.buf[row * m.cols + col] = val;
}

/**
 * Function: matrix_new
 * Description: Creates a new matrix with the specified number of rows and columns,
 *              and initializes all elements to zero.
 *
 * Parameters:
 *   rows: The number of rows in the new matrix.
 *   cols: The number of columns in the new matrix.
 *
 * Returns:
 *   The newly created matrix.
 */
matrix matrix_new(int rows, int cols);

/**
 * Function: matrix_from_buf
 * Description: Creates a new matrix using the given buffer of data, rows, and columns.
 *
 * Parameters:
 *   buf: Pointer to the buffer containing the matrix data.
 *   rows: The number of rows in the new matrix.
 *   cols: The number of columns in the new matrix.
 *
 * Returns:
 *   The newly created matrix using the provided buffer.
 */
matrix matrix_from_buf(double *buf, int rows, int cols);

/**
 * Function: matrix_del
 * Description: Deallocates the memory associated with the given matrix.
 *
 * Parameters:
 *   m: The matrix to be deallocated.
 */
void matrix_del(matrix m);

/**
 * Function: matrix_dot
 * Description: Performs the matrix dot product (matrix multiplication) of two matrices.
 *
 * Parameters:
 *   out: The output matrix to store the result of the dot product.
 *   m1: The first matrix operand.
 *   m2: The second matrix operand.
 */
void matrix_dot(matrix out, const matrix m1, const matrix m2);

/**
 * Function: matrix_print
 * Description: Prints the elements of the matrix to the console.
 *
 * Parameters:
 *   m: The matrix to be printed.
 */
void matrix_print(matrix m);

/**
 * Function: matrix_get_row
 * Description: Extracts a row from the given matrix and returns it as a new matrix.
 *
 * Parameters:
 *   m: The input matrix.
 *   row: The index of the row to be extracted.
 *
 * Returns:
 *   A new matrix representing the extracted row.
 */
matrix matrix_get_row(matrix m, int row);

/**
 * Function: matrix_scalar_multiply
 * Description: Multiplies each element of the matrix by the given scalar value
 *              and stores the result in the output matrix.
 *
 * Parameters:
 *   out: The output matrix to store the result.
 *   m: The input matrix to be multiplied.
 *   x: The scalar value to be multiplied with each element of the matrix.
 */
void matrix_scalar_multiply(matrix out, const matrix m, double x);

/**
 * Function: matrix_scalar_multiply_ip
 * Description: Multiplies each element of the matrix by the given scalar value
 *              in place, modifying the original matrix.
 *
 * Parameters:
 *   m: The input matrix to be multiplied.
 *   x: The scalar value to be multiplied with each element of the matrix.
 */
void matrix_scalar_multiply_ip(matrix m, double x);

/**
 * Function: matrix_add_ip
 * Description: Adds the elements of the second matrix to the corresponding elements
 *              of the first matrix, modifying the first matrix in place.
 *
 * Parameters:
 *   m1: The first matrix.
 *   m2: The second matrix to be added to the first matrix.
 */
void matrix_add_ip(matrix m1, matrix m2);

/**
 * Function: matrix_add_ip_T
 * Description: Adds the transpose of the second matrix to the first matrix,
 *              modifying the first matrix in place.
 *
 * Parameters:
 *   m1: The first matrix.
 *   m2: The second matrix whose transpose will be added to the first matrix.
 */
void matrix_add_ip_T(matrix m1, matrix m2);

/**
 * Function: matrix_print_dims
 * Description: Prints the dimensions (rows, columns) of the matrix to the console.
 *
 * Parameters:
 *   m: The matrix whose dimensions are to be printed.
 */
void matrix_print_dims(matrix m);

#endif  // MATRIX_H
