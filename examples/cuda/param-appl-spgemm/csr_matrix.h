#pragma once
#include "coo_matrix.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct csr_matrix_tuple {
    int   col;
    float val;
} csr_matrix_tuple_t;

typedef struct csr_matrix {
    int  n;
    int  nnz;
    int  *rowptrs;
    csr_matrix_tuple_t *nonzeros;
} csr_matrix_t;

int  csr_matrix_init_from_coo(csr_matrix_t *csr, coo_matrix_t *coo);
void csr_matrix_dest(csr_matrix_t *csr);
#ifdef __cplusplus
}
#endif
