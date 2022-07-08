#include "bsg_manycore.h"
#include "bsg_manycore_atomic.h"
#include "bsg_tile_config_vars.h"
#include "bsg_cuda_lite_barrier.h"
#include "csr_matrix.h"
#include "appl.hpp"
#include "tuple_dyn_vec.h"

static int floor_log2(int x)
{
    int i = -1;
    int j = i+1;
    while  ((1 << j) < x) {
        i = j;
        j = j+1;
    }
    return i;
}

static int ceil_log2(int x)
{
    return i+1;
}

static int tree_rchild(int root)
{
    return 2*root + 2;
}

static int tree_lchild(int root)
{
    return 2*root + 1;
}

static void parallel_scan(
    int  *in
    ,int *out
    ,int  n
    ) {
    int *tree = appl::malloc(sizeof(int)*2*appl::get_nthreads());

    appl::parallel_for(0, appl::get_nthreads(), [=](int tid){
            // calculate range
            int region_size = (n+appl::get_nthreads())/appl::get_nthreads();
            int start = tid * region_size;
            int end   = std::min(start + region_size, n+1);

            // calculate local sum
            int sum = 0;
            for (int i = start; i < end; i++) {
                out[i] = sum;
                sum += in[i];
            }

            // update tree
            int r = 0;
            int m = appl::get_nthreads();
            int L = ceil_log2(m);
            for (int l = 0; l < L; l++) {
                bsg_amoadd(&tree[r], sum);
                m >>= 1;
                if (m & tid) {
                    r = tree_rchild(r);
                } else {
                    r = tree_lchild(r);
                }
            }
            bsg_amoadd(&tree[r], sum);
        });

    // sync
    
    appl::parallel_for(0, appl::get_nthreads(), [=](int tid){
            // calculate range
            int region_size = (n+appl::get_nthreads())/appl::get_nthreads();
            int start = tid * region_size;
            int end   = std::min(start + region_size, n+1);

            // accumulate from sum tree
            int s = 0;
            int r = 0;
            int m = appl::get_nthreads();
            int L = ceil_log2(m);
            
            for (int l = 0; l < L; l++) {
                m >>= 1;
                if (tid & m) {
                    s += tree[tree_lchild(r)];
                    r = tree_rchild(r);
                } else {
                    r = tree_lchild(r);
                }
            }

            // calculate local sum
            for (int i = start; i < end; i++) {
                bsg_amoadd(&out[i], s);
            }
        });
}

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
    tuple_dyn_vec_init(&Bv, nnz);
    Bv.size = nnz;

    // stall on off
    csr_matrix_tuple_t *nonzeros = &B->nonzeros[Bi_off];

    for (int nz = 0; nz < Bi_nnz; nz++) {
        float Bij, Cij;
        int   Bj;
        Bij = nonzeros[nz].val;
        Bj  = nonzeros[nz].col;
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

    // for each nonzero entry in row A[i:]
    for (int nz = 0; nz < nnz; nz++) {
        int Bi    = nonzeros[nz].col;
        float Aij = nonzeros[nz].val;
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
    csr_matrix_t         *A
    ,csr_matrix_t        *B
    ,csr_matrix_t        *C
    ,int                 *C_row_nnz
    ,csr_matrix_tuple_t **C_tmp
    ) {
    appl::runtime_init(dram_buffer);
    appl::sync();
    bsg_cuda_print_stats_kernel_start();
    if (__bsg_id == 0) {
        // 1. solve for each row
        appl::parallel_for(0, A->n, [=](int Ci){
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
                    ,Ci_nnz
                    ,Ci_off
                    );
            });
        
        // 2. scan
        parallel_scan(C_row_nnz, C->rowptrs, C->n);
        
        // 3. copy
        csr_matrix_tuple_t *C_nonzeros = appl::malloc(sizeof(csr_matrix_tuple_t) * C->nnz);
        C->nonzeros = C_nonzeros;
        
        appl::parallel_for(0, A->n, [=](int Ci){
                csr_matrix_tuple_t *src, *dst;
                src = C_tmp[C_i];
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
    bsg_cuda_print_stats_kernel_end();
    bsg_fence();
    appl::sync();
    return;
}
