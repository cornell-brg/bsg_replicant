#pragma once
#include "bsg_manycore_cuda.h"

typedef struct dev_csr_matrix {
    int n;
    int nnz;
    hb_mc_eva_t rowptrs;
    hb_mc_eva_t nonzeros;
} dev_csr_matrix_t;

