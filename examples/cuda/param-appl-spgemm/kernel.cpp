#include <algorithm>
#include "bsg_manycore.h"
#include "bsg_manycore_atomic.h"
#include "bsg_tile_config_vars.h"
#include "bsg_cuda_lite_barrier.h"
#include "csr_matrix.h"
#include "appl.hpp"
#include "parallel_prefix_sum.hpp"
#include "tuple_dyn_vec.h"
#define DEBUG_SPGEMM
#ifdef  DEBUG_SPGEMM
#define bsg_print_int_dbg(x)                    \
    bsg_print_int(x)
#else
#define bsg_print_int_dbg(x)
#endif

#ifdef APPL_IMPL_CELLO
#define spgemm_malloc(size)                     \
    cello_malloc(size)
#else
#define spgemm_malloc(size)                     \
    appl::appl_malloc(size)
#endif

/**
 * perform Aik * B[k;] and update C[i;]
 */
static void scalar_row_product(
    // kernel arguments
    csr_matrix_t        *A
    ,csr_matrix_t       *B
    ,csr_matrix_t       *C
    ,int                *C_row_nnz
    ,csr_matrix_tuple_t**C_tmp
    // row parameters
    ,float            Aij
    ,int              Bi
    ,tuple_dyn_vec_t *Cv
    ) {
    // fetch row data
    int Bi_off = B->rowptrs[Bi];
    int Bi_nnz = B->rowptrs[Bi+1]-Bi_off;

    // initialize a partial buffer
    tuple_dyn_vec_t Bv;
    tuple_dyn_vec_init(&Bv, Bi_nnz);
    Bv.size = Bi_nnz;

    // stall on off
    csr_matrix_tuple_t *nonzeros = &B->nonzeros[Bi_off];
    bsg_print_int_dbg(4000000+Bi_nnz);
    for (int nz = 0; nz < Bi_nnz; nz++) {
        float Bij, Cij;
        int   Bj;
        Bij = nonzeros[nz].val;
        Bj  = nonzeros[nz].col;
        bsg_print_int_dbg(5000000+Bj);
        Cij = Aij * Bij;
        Bv.v[nz].col =  Bj;
        Bv.v[nz].val = Cij;
    }

    // merge results
    tuple_dyn_vec_merge(Cv, &Bv, Cv);
}

static void solve_row(
    // kernel arguments
    csr_matrix_t        *A
    ,csr_matrix_t       *B
    ,csr_matrix_t       *C
    ,int                *C_row_nnz
    ,csr_matrix_tuple_t**C_tmp
    // row parameters
    ,int Ci
    ,int Ci_off
    ,int Ci_nnz
    ) {

    // fetch row meta data
    int off = Ci_off;
    int nnz = Ci_nnz;
    csr_matrix_tuple_t *nonzeros = &A->nonzeros[off];

    tuple_dyn_vec_t Cv;
    tuple_dyn_vec_init(&Cv, 0);
    bsg_print_int_dbg(2000000 + nnz);
    // for each nonzero entry in row A[i:]
    for (int nz = 0; nz < nnz; nz++) {        
        int Bi    = nonzeros[nz].col;
        float Aij = nonzeros[nz].val;
        bsg_print_int_dbg(3000000 + Bi);
        scalar_row_product(
             A
            ,B
            ,C
            ,C_row_nnz
            ,C_tmp
            ,Aij
            ,Bi
            ,&Cv
            );
    }

    // update the global number of nonzeros
    bsg_amoadd(&C->nnz, Cv.size);
    C_row_nnz[Ci] = Cv.size;
    C_tmp[Ci] = Cv.v;
}

extern "C" void spgemm(
    // kernel arguments
    csr_matrix_t         *A
    ,csr_matrix_t        *B
    ,csr_matrix_t        *C
    ,int                 *C_row_nnz
    ,csr_matrix_tuple_t **C_tmp
    // appl runtime
    ,int *dram_buffer
    ) {

    appl::runtime_init(dram_buffer);
    tuple_vec_libinit();
    
    appl::sync();
    bsg_cuda_print_stat_kernel_start();

    if (__bsg_id == 0) {
        // 1. solve for each row
        appl::parallel_for(0, A->n, [=](int Ci){
                bsg_print_int_dbg(1000000 + Ci);
                // fetch A_i row data
                int Ci_off = A->rowptrs[Ci];
                int Ci_nnz = A->rowptrs[Ci+1] - Ci_off;
                // solve for row Ci
                solve_row(
                     A
                    ,B
                    ,C
                    ,C_row_nnz
                    ,C_tmp
                    ,Ci
                    ,Ci_off
                    ,Ci_nnz
                    );
            });
        
        // 2. scan
        parallel_prefix_sum(C_row_nnz, C->rowptrs, C->n);
        
        // 3. copy
        csr_matrix_tuple_t *C_nonzeros = (csr_matrix_tuple_t*)spgemm_malloc(
            sizeof(csr_matrix_tuple_t) * C->nnz
            );
        C->nonzeros = C_nonzeros;
        
        appl::parallel_for(0, A->n, [=](int Ci){
                csr_matrix_tuple_t *src, *dst;
                src = C_tmp[Ci];
                int Ci_off = C->rowptrs[Ci];
                int Ci_nnz = C->rowptrs[Ci+1]-Ci_off;
                dst = &C->nonzeros[Ci_off];
                for (int nz = 0; nz < Ci_nnz; nz++) {
                    dst[nz] = src[nz];
                }
            });        
    } else {
        appl::worker_thread_init();
    }
    appl::runtime_end();
    bsg_cuda_print_stat_kernel_end();
    bsg_fence();
    appl::sync();
    return;
}
