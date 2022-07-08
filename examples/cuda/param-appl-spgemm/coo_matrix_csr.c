#include "utils.h"
#include "coo_matrix_csr.h"
int  coo_matrix_init_from_csr(coo_matrix_t *coo, csr_matrix_t *csr)
{
    // set meta
    coo->n  = csr->n;
    coo->nz = csr->nnz;

    // allocate memory
    coo_matrix_tuple_t *nonzeros;
    BSG_CUDA_CALL(try_malloc(sizeof(coo_matrix_tuple_t)*coo->nz, (void**)&nonzeros));

    // copy row-by-row
    int dnz = 0;    
    for (int i = 0; i < csr->n; i++) {
        int off = csr->rowptrs[i];
        int nnz = csr->rowptrs[i+1]-off;
        csr_matrix_tuple_t *row_nonzeros = &csr->nonzeros[off];
        for (int nz = 0; nz < nnz; nz++) {
            nonzeros[dnz].row = i;
            nonzeros[dnz].col = row_nonzeros[nz].col;
            nonzeros[dnz].val = row_nonzeros[nz].val;
        }
    }
    coo->nonzeros = nonzeros;

    return HB_MC_SUCCESS;
}
