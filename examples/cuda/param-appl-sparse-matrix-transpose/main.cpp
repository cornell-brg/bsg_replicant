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

#define MAX_WORKERS 128
#define HB_L2_CACHE_LINE_WORDS 16
#define BUF_FACTOR 16385
#define BUF_SIZE (MAX_WORKERS * HB_L2_CACHE_LINE_WORDS * BUF_FACTOR)

hb_mc_device_t dev;

static
int dev_csr_matrix_init(csr_matrix_t *csr, hb_mc_eva_t *csr_dev_ptr)
{
    // set metadata
    dev_csr_matrix_t _, *dev_csr = &_;
    hb_mc_eva_t ptr;

    dev_csr->n = csr->n;
    dev_csr->nnz = csr->nnz;
    BSG_CUDA_CALL(hb_mc_device_malloc(&dev, (dev_csr->n+1) * sizeof(int), &dev_csr->rowptrs));
    BSG_CUDA_CALL(hb_mc_device_malloc(&dev, dev_csr->nnz * sizeof(csr_matrix_tuple_t), &dev_csr->nonzeros));
    BSG_CUDA_CALL(hb_mc_device_malloc(&dev, sizeof(dev_csr_matrix_t), &ptr));

    // copy nonzeros and rowptrs
    hb_mc_dma_htod_t htod [] = {
        { dev_csr->rowptrs, csr->rowptrs, (dev_csr->n+1) * sizeof(int) },
        { dev_csr->nonzeros, csr->nonzeros, dev_csr->nnz * sizeof(csr_matrix_tuple_t) },
        { ptr, dev_csr, sizeof(dev_csr_matrix_t) }
    };
    BSG_CUDA_CALL(hb_mc_device_dma_to_device(&dev, htod, 3));

    *csr_dev_ptr = ptr;
    return HB_MC_SUCCESS;
}

static
int dev_csr_matrix_init_empty(const csr_matrix_t *csr, hb_mc_eva_t *csr_dev_ptr)
{
    // set metadata
    dev_csr_matrix_t _, *dev_csr = &_;
    hb_mc_eva_t ptr;

    dev_csr->n   = csr->n;
    dev_csr->nnz = csr->nnz;
    BSG_CUDA_CALL(hb_mc_device_malloc(&dev, (dev_csr->n+1)*sizeof(int), &dev_csr->rowptrs));
    BSG_CUDA_CALL(hb_mc_device_malloc(&dev, (dev_csr->nnz)*sizeof(csr_matrix_tuple_t), &dev_csr->nonzeros));
    BSG_CUDA_CALL(hb_mc_device_malloc(&dev, sizeof(dev_csr_matrix_t), &ptr));

    // copy nonzeros and rowptrs
    hb_mc_dma_htod_t htod [] = {
        { ptr, dev_csr, sizeof(dev_csr_matrix_t) }
    };
    BSG_CUDA_CALL(hb_mc_device_dma_to_device(&dev, htod, 1));

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

    printf("updating matrix: csr->n = %d, csr->nnz = %d\n"
           ,csr->n
           ,csr->nnz
        );

    BSG_CUDA_CALL(try_malloc(sizeof(int)*(csr->n+1), (void**)&csr->rowptrs));
    BSG_CUDA_CALL(try_malloc(sizeof(csr_matrix_tuple_t)*csr->nnz, (void**)&csr->nonzeros));

    // read nonzeros and row pointers
    hb_mc_dma_dtoh_t dtoh [] = {
        { dev_csr->rowptrs, csr->rowptrs, (csr->n+1)*sizeof(int)},
        { dev_csr->nonzeros, csr->nonzeros, csr->nnz*sizeof(csr_matrix_tuple_t) }
    };

    BSG_CUDA_CALL(hb_mc_device_dma_to_host(&dev, dtoh, 2));

    return HB_MC_SUCCESS;
}

int SparseMatrixTransposeMain(int argc, char *argv[])
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
    coo_matrix_t I_coo;
    csr_matrix_t I_csr;
    BSG_CUDA_CALL(coo_matrix_init_from_mm(
                      &I_coo
                      ,f
                      ));
    BSG_CUDA_CALL(csr_matrix_init_from_coo(
                      &I_csr
                      ,&I_coo
                      ));

    // initialize device
    BSG_CUDA_CALL(hb_mc_device_init(
                      &dev
                      ,"cuda"
                      ,0
                      ));
    BSG_CUDA_CALL(hb_mc_device_program_init(
                      &dev
                      ,kernels
                      ,"cuda"
                      ,0
                      ));

    // initialize input (I) and output (O)
    hb_mc_eva_t I_dev, O_dev;
    BSG_CUDA_CALL(dev_csr_matrix_init(
                      &I_csr
                      ,&I_dev
                      ));
    BSG_CUDA_CALL(dev_csr_matrix_init_empty(
                      &I_csr
                      ,&O_dev
                      ));

    // allocate dram buffer
    hb_mc_eva_t dram_buffer_dev;
    BSG_CUDA_CALL(hb_mc_device_malloc(
                      &dev
                      ,BUF_SIZE * sizeof(uint32_t)
                      ,&dram_buffer_dev
                      ));

    int kargc = 3;
    hb_mc_eva_t kargv [] = {I_dev, O_dev, dram_buffer_dev};
    hb_mc_dimension_t gd = {1 , 1};
    hb_mc_dimension_t tg = {bsg_tiles_X , bsg_tiles_Y};
    printf("launching sparse transpose kernel\n");
    BSG_CUDA_CALL(hb_mc_kernel_enqueue(
                      &dev
                      ,gd
                      ,tg
                      ,"sparse_transpose"
                      ,kargc
                      ,kargv
                      ));
    
    BSG_CUDA_CALL(hb_mc_device_tile_groups_execute(&dev));
    printf("sparse transpose kernel complete!\n");

    // get results
    csr_matrix_t O_csr;
    BSG_CUDA_CALL(csr_matrix_update_from_dev(
                      &O_csr
                      ,O_dev
                      ));

    // check the result with eigen
    printf("initializing I_eigen and O_eigen\n");
    eigen_sparse_matrix_t I_eigen
        = eigen_sparse_matrix_from_coo(&I_coo);
    eigen_sparse_matrix_t O_eigen
        = eigen_sparse_matrix_t(I_eigen.transpose());

    coo_matrix_t O_coo;
    printf("initializing O_eigen_device\n");
    BSG_CUDA_CALL(coo_matrix_init_from_csr(
                      &O_coo
                      ,&O_csr
                      ));
    eigen_sparse_matrix_t O_eigen_device
        = eigen_sparse_matrix_from_coo(&O_coo);

    // compare eigen vs device
    int r = HB_MC_SUCCESS;
    printf("checking output against eigen\n");
    if (!eigen_sparse_matrices_are_equal(O_eigen, O_eigen_device)) {
        r = HB_MC_FAIL;
    }
    printf("%s result\n", r == HB_MC_SUCCESS ? "correct" : "incorrect");
    BSG_CUDA_CALL(hb_mc_device_program_finish(&dev));
    BSG_CUDA_CALL(hb_mc_device_finish(&dev));

    csr_matrix_dest(&I_csr);
    coo_matrix_dest(&I_coo);
    csr_matrix_dest(&O_csr);
    coo_matrix_dest(&O_coo);

    return r;
}
declare_program_main("APPL Parameterized Sparse Matrix Transpose", SparseMatrixTransposeMain);
