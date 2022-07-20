#include <bsg_manycore_tile.h>
#include <bsg_manycore_errno.h>
#include <bsg_manycore_tile.h>
#include <bsg_manycore_loader.h>
#include <bsg_manycore_cuda.h>
#include <bsg_manycore_regression.h>
#include "eigen_sparse_matrix.hpp"
#include "Eigen/Dense"
#include "csr_matrix.h"
#include "coo_matrix_mmio.h"
#include "coo_matrix_csr.h"
#include "dev_csr_matrix.h"
#include "utils.h"
#include <vector>
#include <random>

typedef Eigen::VectorXf eigen_vector_t;

hb_mc_device_t dev;

std::vector<hb_mc_dma_htod_t> htod;

static
std::vector<float> random_vector(int n)
{
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dis (0.0, 1.0);
    std::vector<float> v(n);
    for (int i = 0; i < n; i++) {
        v[i] = dis(gen);
    }
    return v;
}

static
int dev_csr_matrix_init(csr_matrix_t *csr, hb_mc_eva_t *csr_dev_ptr)
{
    // set metadata
    dev_csr_matrix_t *dev_csr;
    BSG_CUDA_CALL(try_malloc(sizeof(*dev_csr), (void**)&dev_csr));

    hb_mc_eva_t ptr;
    dev_csr->n = csr->n;
    dev_csr->nnz = csr->nnz;
    BSG_CUDA_CALL(hb_mc_device_malloc(
                      &dev
                      ,(dev_csr->n+1) * sizeof(int)
                      ,&dev_csr->rowptrs
                      ));
    BSG_CUDA_CALL(hb_mc_device_malloc(
                      &dev
                      ,dev_csr->nnz * sizeof(csr_matrix_tuple_t)
                      ,&dev_csr->nonzeros
                      ));
    BSG_CUDA_CALL(hb_mc_device_malloc(
                      &dev
                      ,sizeof(dev_csr_matrix_t)
                      ,&ptr
                      ));

    // copy nonzeros and rowptrs
    htod.push_back({dev_csr->rowptrs, csr->rowptrs, (dev_csr->n+1)*sizeof(int)});
    htod.push_back({dev_csr->nonzeros, csr->nonzeros, dev_csr->nnz * sizeof(csr_matrix_tuple_t)});
    htod.push_back({ptr, dev_csr, sizeof(*dev_csr)});

    *csr_dev_ptr = ptr;
    return HB_MC_SUCCESS;
}

int SpMVMain(int argc, char *argv[])
{
    char *kernels = argv[1];
    char *mtx_file = argv[3];
    FILE *f = fopen(mtx_file, "r");

    printf("Input matrix: %s\n", mtx_file);
    if (!f) {
        fprintf(stderr, "failed to open '%s': %m\n", mtx_file);
        return HB_MC_FAIL;
    }

    // construct csr/coo
    coo_matrix_t A_coo;
    BSG_CUDA_CALL(coo_matrix_init_from_mm(&A_coo, f));

    csr_matrix_t A_csr;
    BSG_CUDA_CALL(csr_matrix_init_from_coo(&A_csr, &A_coo));

    // initialize device
    BSG_CUDA_CALL(hb_mc_device_init(&dev, "cuda", 0));
    BSG_CUDA_CALL(hb_mc_device_program_init(&dev, kernels, "cuda", 0));

    // initialize input (A)
    hb_mc_eva_t A_dev;
    BSG_CUDA_CALL(dev_csr_matrix_init(
                      &A_csr
                      ,&A_dev
                      ));

    // initialize input  (X)
    hb_mc_eva_t X_dev;
    BSG_CUDA_CALL(hb_mc_device_malloc(&dev, sizeof(float)*A_csr.n, &X_dev));
    std::vector<float> X = random_vector(A_csr.n);
    htod.push_back({X_dev, X.data(), sizeof(float)*X.size()});

    // initialize output (Y)
    hb_mc_eva_t Y_dev;
    BSG_CUDA_CALL(hb_mc_device_malloc(&dev, sizeof(float)*A_csr.n, &Y_dev));
    std::vector<float> Y (A_csr.n, 0.0);
    htod.push_back({Y_dev, Y.data(), sizeof(float)*Y.size()});


    // allocate dram buffer
    hb_mc_eva_t dram_buffer_dev;
    BSG_CUDA_CALL(hb_mc_device_malloc(
                      &dev
                      ,128*1024*1024
                      ,&dram_buffer_dev
                      ));

    // dma
    BSG_CUDA_CALL(hb_mc_device_dma_to_device(&dev, htod.data(), htod.size()));

    std::vector<hb_mc_eva_t> kargv = {
        A_dev, X_dev, Y_dev, dram_buffer_dev
    };
    int kargc = kargv.size();

    hb_mc_dimension_t gd = {1 , 1};
    hb_mc_dimension_t tg = {bsg_tiles_X , bsg_tiles_Y};
    printf("launching spmv kernel\n");
    BSG_CUDA_CALL(hb_mc_kernel_enqueue(
                      &dev
                      ,gd
                      ,tg
                      ,"spmv"
                      ,kargc
                      ,kargv.data()
                      ));

    BSG_CUDA_CALL(hb_mc_device_tile_groups_execute(&dev));
    printf("spmv kernel complete!\n");

    // get results
    hb_mc_dma_dtoh_t dtoh [] = {
        { Y_dev, Y.data(), sizeof(float)*Y.size() }
    };
    BSG_CUDA_CALL(hb_mc_device_dma_to_host(&dev, dtoh, 1));

    // check the result with eigen
    printf("initializing A_eigen\n");

    eigen_sparse_matrix_t A_eigen
        = eigen_sparse_matrix_from_coo(&A_coo);

    Eigen::Map<eigen_vector_t> X_eigen(X.data(), X.size());
    Eigen::Map<eigen_vector_t> Y_eigen_dev(Y.data(), Y.size());
    eigen_vector_t Y_eigen = A_eigen * X_eigen;

    #define fmt "15.10f"
    for (int i = 0; i < Y_eigen_dev.size(); i++) {
        if (Y_eigen[i] != Y_eigen_dev[i]) {
            printf("mismatch: Y_eigen[%7d] = %"fmt", Y_eigen_dev[%7d] = %"fmt"\n"
                   ,i
                   ,Y_eigen[i]
                   ,i
                   ,Y_eigen_dev[i]
                );
        }
    }

    float norm = (Y_eigen-Y_eigen_dev).norm();
    printf("(Y_eigen - Y_eigen_dev).norm() = %f\n"
           ,norm
        );
    int r = (norm < 1e-3 ? HB_MC_SUCCESS : HB_MC_FAIL);

    BSG_CUDA_CALL(hb_mc_device_program_finish(&dev));
    BSG_CUDA_CALL(hb_mc_device_finish(&dev));

    csr_matrix_dest(&A_csr);
    coo_matrix_dest(&A_coo);

    return r;
}

declare_program_main("APPL SpMV", SpMVMain);
