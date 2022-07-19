#include "bsg_manycore.h"
#include "appl.hpp"
#include "csr_matrix.h"
#include <cmath>

#define DEBUG_SPMV
#ifdef  DEBUG_SPMV
#define spmv_print_int(x)                       \
    bsg_print_int(x)
#else
#define spmv_print_int(x)
#endif

#define DBG_SPMV_ROW  100000000
#define DBG_SPMV_RNNZ 200000000
#define DBG_SPMV_COL  300000000

extern "C" int spmv(
    csr_matrix_t *A,
    const float *X,
    float *Y,
    int *dram_buffer
    )
{
    appl::runtime_init(dram_buffer);
    appl::sync();
    bsg_cuda_print_stat_kernel_start();

    if (__bsg_id == 0) {
        appl::parallel_for(0, A->n, [=](int i) {
                spmv_print_int(DBG_SPMV_ROW + i);
                // offset and nnz
                int start = A->rowptrs[i];
                int stop  = A->rowptrs[i+1];
                float v_Y = 0.0;
                spmv_print_int(DBG_SPMV_RNNZ + (stop-start));
                // an optimization would unroll this loop
                for (int nz = start; nz < stop; nz++) {
                    // fetch column and values
                    int j     = A->nonzeros[nz].col;
                    float v_A = A->nonzeros[nz].val;
                    // fetch row from X
                    float v_X = X[j];
                    // fma
                    v_Y = fmaf(v_A, v_X, v_Y);
                    spmv_print_int(DBG_SPMV_COL + j);
                }
                Y[i] = v_Y;
                //Y[i] = 1.0;
            });
    } else {
        appl::worker_thread_init();
    }

    appl::runtime_end();
    bsg_cuda_print_stat_kernel_end();
    bsg_fence();
    appl::sync();

    return 0;
}
