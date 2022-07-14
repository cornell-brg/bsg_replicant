#include "utils.h"
#include "coo_matrix_mmio.h"

#define MMIO_CALL(stmt)                                         \
    {                                                           \
        int __r = stmt;                                         \
        if (__r != 0) {                                         \
            fprintf(stderr, "'%s' failed\n", #stmt);            \
            return HB_MC_FAIL;                                  \
        }                                                       \
    }

int coo_matrix_init_from_mm(coo_matrix_t *mtx, FILE *mtx_file)
{

    /* read the matrix header */
    MM_typecode matcode;
    int M, N, nz;
    MMIO_CALL(mm_read_banner(mtx_file, &matcode));
    if (!mm_is_sparse(matcode)) {
        fprintf(stderr, "failed to parse matrix: Requries sparse matrix\n");
        return HB_MC_FAIL;        
    }    
    MMIO_CALL(mm_read_mtx_crd_size(mtx_file, &M, &N, &nz));
    printf("Reading sparse matrix size (%d, %d) with %d non-zeros\n"
           ,M
           ,N
           ,nz);
    if (M != N) {
        fprintf(stderr, "failed to parse: Requires square matrix\n");
        return HB_MC_FAIL;
    }

    mtx->n = N;
    mtx->nz = nz;

    coo_matrix_tuple_t *nonzeros;
    BSG_CUDA_CALL(try_malloc(sizeof(coo_matrix_tuple_t)*mtx->nz, (void**)&nonzeros));

    for (int nz = 0; nz < mtx->nz; nz++) {
        int i, j;
        float v;
        fscanf(mtx_file, "%d %d %f", &i, &j, &v);
        nonzeros[nz].row = i-1;
        nonzeros[nz].col = j-1;
        nonzeros[nz].val = v;
    }
    mtx->nonzeros = nonzeros;
    
    return HB_MC_SUCCESS;
}

void coo_matrix_dest(coo_matrix_t *mtx)
{
    free(mtx->nonzeros);
    mtx->nonzeros = 0;
    mtx->n = 0;
    mtx->nz = 0;
}
