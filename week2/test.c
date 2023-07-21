#include "matrix.h"

int
main()
{
    matrix m = matrix_new(1, 100);
    for (int j = 0; j < 100; j++) matrix_set(m, 0, j, (double)(2 * j));

    matrix m2 = matrix_new(100, 1);
    for (int j = 0; j < 100; j++) matrix_set(m2, j, 0, (double)(j));

    matrix_print(m);
    matrix_print(m2);

    matrix p = matrix_dot(m, m2);
    matrix_print(p);

    matrix_del(m);
    matrix_del(m2);
    matrix_del(p);
}
