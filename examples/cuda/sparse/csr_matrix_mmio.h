#pragma once
#include "coo_matrix.h"
#include "csr_matrix.h"
#include "coo_matrix_mmio.h"
#include "bsg_manycore_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

inline int  csr_matrix_init_from_mmio(csr_matrix_t *csr, FILE *mmfile)
{
    coo_matrix_t coo;
    BSG_CUDA_CALL(coo_matrix_init_from_mm(&coo, mmfile));
    BSG_CUDA_CALL(csr_matrix_init_from_coo(csr, &coo));
    csr_matrix_dest(&coo);
    return HB_MC_SUCCESS;
}

#ifdef __cplusplus
}
#endif
