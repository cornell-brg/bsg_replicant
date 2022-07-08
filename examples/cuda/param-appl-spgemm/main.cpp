#include <bsg_manycore_tile.h>
#include <bsg_manycore_errno.h>
#include <bsg_manycore_tile.h>
#include <bsg_manycore_loader.h>
#include <bsg_manycore_cuda.h>
#include <bsg_manycore_regression.h>
#include "eigen_sparse_matrix.hpp"
#include "csr_matrix.h"
#include "coo_matrix_mmio.h"
#include "coo_matrix_csr.h"
#include "dev_csr_matrix.h"
#include "utils.h"

hb_mc_device_t dev;

static
int dev_csr_matrix_init(csr_matrix_t *csr, hb_mc_eva_t *csr_dev_ptr)
{
    // set metadata
    dev_csr_matrix_t _, *dev_csr = &_;
    hb_mc_eva_t ptr;

    dev_csr->n = csr->n;
    dev_csr->nnz = csr->nnz;
    BSG_CUDA_CALL(hb_mc_device_malloc(&dev, dev_csr->n * sizeof(int), &dev_csr->rowptrs));
    BSG_CUDA_CALL(hb_mc_device_malloc(&dev, dev_csr->nnz * sizeof(csr_matrix_tuple_t), &dev_csr->nonzeros));
    BSG_CUDA_CALL(hb_mc_device_malloc(&dev, sizeof(dev_csr_matrix_t), &ptr));

    // copy nonzeros and rowptrs
    hb_mc_dma_htod_t htod [] = {
        { dev_csr->rowptrs, csr->rowptrs, dev_csr->n * sizeof(int) },
        { dev_csr->nonzeros, csr->nonzeros, dev_csr->nnz * sizeof(csr_matrix_tuple_t) },
        { ptr, dev_csr, sizeof(dev_csr_matrix_t) }
    };
    BSG_CUDA_CALL(hb_mc_device_dma_to_device(&dev, htod, 3));

    *csr_dev_ptr = ptr;
    return HB_MC_SUCCESS;
}

static
int dev_csr_matrix_init_empty(csr_matrix_t *csr, hb_mc_eva_t *csr_dev_ptr)
{
    // set metadata
    dev_csr_matrix_t _, *dev_csr = &_;
    hb_mc_eva_t ptr;

    dev_csr->n = csr->n;
    dev_csr->nnz = 0;
    BSG_CUDA_CALL(hb_mc_device_malloc(&dev, dev_csr->n * sizeof(int), &dev_csr->rowptrs));
    BSG_CUDA_CALL(hb_mc_device_malloc(&dev, sizeof(dev_csr_matrix_t), &ptr));

    // copy nonzeros and rowptrs
    hb_mc_dma_htod_t htod [] = {
        { dev_csr->rowptrs, csr->rowptrs, dev_csr->n * sizeof(int) },
        { ptr, dev_csr, sizeof(dev_csr_matrix_t) }
    };
    BSG_CUDA_CALL(hb_mc_device_dma_to_device(&dev, htod, 2));

    *csr_dev_ptr = ptr;
    return HB_MC_SUCCESS;
}

static
int csr_matrix_update_from_dev(csr_matrix_t *csr, hb_mc_eva_t csr_dev_ptr)
{
    dev_csr_matrix_t _, *dev_csr = &_;

    // read header
    BSG_CUDA_CALL(hb_mc_device_memcpy_to_host(
                      &dev
                      ,dev_csr
                      ,csr_dev_ptr
                      ,sizeof(dev_csr_matrix_t)
                      ));

    csr->n   = dev_csr->n;
    csr->nnz = dev_csr->nnz;
    BSG_CUDA_CALL(try_malloc(sizeof(int)*csr->n, (void**)&csr->rowptrs));
    BSG_CUDA_CALL(try_malloc(sizeof(csr_matrix_tuple_t)*csr->nnz, (void**)&csr->nonzeros));

    // read nonzeros and row pointers
    hb_mc_dma_dtoh_t dtoh [] = {
        { dev_csr->rowptrs, csr->rowptrs, csr->n*sizeof(int)},
        { dev_csr->nonzeros, csr->nonzeros, csr->nnz*sizeof(csr_matrix_tuple_t) }
    };

    BSG_CUDA_CALL(hb_mc_device_dma_to_host(&dev, dtoh, 2));

    return HB_MC_SUCCESS;
}

int SpGEMMMain(int argc, char *argv[])
{
    char *kernels = argv[1];
    char *mtx_file = argv[3];
    FILE *f = fopen(mtx_file, "r");

    printf("Input matrix: %s\n", mtx_file);
    if (!f) {
        fprintf(stderr, "failed to open '%s': %m\n", mtx_file);
        return HB_MC_FAIL;
    }

    // construct csr
    coo_matrix_t coo;
    csr_matrix_t csr;
    coo_matrix_init_from_mm(&coo, f);
    csr_matrix_init_from_coo(&csr, &coo);

    // initialize device
    BSG_CUDA_CALL(hb_mc_device_init(&dev, "cuda", 0));
    BSG_CUDA_CALL(hb_mc_device_program_init(&dev, kernels, "cuda", 0));

    // initialize A, B, and C on device
    hb_mc_eva_t A_dev, B_dev, C_dev;
    BSG_CUDA_CALL(dev_csr_matrix_init(&csr, &A_dev));
    BSG_CUDA_CALL(dev_csr_matrix_init(&csr, &B_dev));
    BSG_CUDA_CALL(dev_csr_matrix_init_empty(&csr, &C_dev));

    hb_mc_eva_t C_row_nnz, C_tmp;
    BSG_CUDA_CALL(hb_mc_device_malloc(&dev, csr.n * sizeof(int), &C_row_nnz));
    BSG_CUDA_CALL(hb_mc_device_malloc(&dev, csr.n * sizeof(hb_mc_eva_t), &C_tmp));

    // allocate dram buffer
    hb_mc_eva_t dram_buffer_dev;
    BSG_CUDA_CALL(hb_mc_device_malloc(&dev, 128*1024*1024, &dram_buffer_dev));
    
    int kargc = 6;
    hb_mc_eva_t kargv [] = {A_dev, B_dev, C_dev, C_row_nnz, C_tmp, dram_buffer_dev};
    hb_mc_dimension_t gd = {1 , 1};
    hb_mc_dimension_t tg = {bsg_tiles_X , bsg_tiles_Y};
    printf("launching spgemm kernel\n");
    BSG_CUDA_CALL(hb_mc_kernel_enqueue(&dev, gd, tg, "spgemm", kargc, kargv));
    BSG_CUDA_CALL(hb_mc_device_tile_groups_execute(&dev));
    printf("spgemm kernel complete!\n");
    
    // get results
    csr_matrix_dest(&csr);
    BSG_CUDA_CALL(csr_matrix_update_from_dev(&csr, C_dev));

    // check the result with eigen
    eigen_sparse_matrix_t eigen_input  = eigen_sparse_matrix_from_coo(&coo);
    eigen_sparse_matrix_t eigen_output = eigen_sparse_matrix_t(eigen_input * eigen_input);

    coo_matrix_dest(&coo);
    coo_matrix_init_from_csr(&coo, &csr);
    eigen_sparse_matrix_t device_output = eigen_sparse_matrix_from_coo(&coo);

    // compare eigen vs device
    int r = HB_MC_SUCCESS;
    if (!eigen_sparse_matrices_are_equal(eigen_output, device_output)) {
        r = HB_MC_FAIL;
    }
    
    BSG_CUDA_CALL(hb_mc_device_program_finish(&dev));
    BSG_CUDA_CALL(hb_mc_device_finish(&dev));
    coo_matrix_dest(&coo);
    csr_matrix_dest(&csr);
    
    return r;
}
declare_program_main("APPL Parameterized SpGEMM", SpGEMMMain);
