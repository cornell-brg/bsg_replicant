#include <bsg_manycore_tile.h>
#include <bsg_manycore_errno.h>
#include <bsg_manycore_tile.h>
#include <bsg_manycore_loader.h>
#include <bsg_manycore_cuda.h>
#include <bsg_manycore_regression.h>
#include "eigen_sparse_matrix.hpp"
#include "csr_matrix.h"
#include "coo_matrix_mmio.h"


int SpGEMMMain(int argc, char *argv[])
{
    char *mtx_file = argv[3];
    FILE *f = fopen(mtx_file, "r");

    printf("Input matrix: %s\n", mtx_file);
    if (!f) {
        fprintf(stderr, "failed to open '%s': %m\n", mtx_file);
        return HB_MC_FAIL;
    }

    coo_matrix_t coo;
    csr_matrix_t csr;
    coo_matrix_init_from_mm(&coo, f);
    csr_matrix_init_from_coo(&csr, &coo);

    eigen_sparse_matrix_t matrix = eigen_sparse_matrix_from_coo(&coo);
    
    coo_matrix_dest(&coo);
    csr_matrix_dest(&csr);
    
    return HB_MC_SUCCESS;
}
declare_program_main("APPL Parameterized SpGEMM", SpGEMMMain);
