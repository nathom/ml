#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"
#include "multivariate_linear_regression.h"

int
main(int argc, char **argv)
{
    if (argc < 2) {
        printf(
            "Usage: %s DEMO\n\n"
            "Available demos: mlr, pr\n",
            argv[0]);
        return 1;
    }

    const char *demo = argv[1];
    if (strcmp("mlr", demo) == 0) {
        mlr_demo();
    } else if (strcmp("pr", demo) == 0) {
        // pr_demo();
    }
}
