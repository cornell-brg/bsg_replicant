#include <algorithm>
#include "csr_matrix.h"
#include "appl.hpp"
#include "parallel_prefix_sum.hpp"

#ifdef  DEBUG_SPARSE_MATRIX_TRANSPOSE
#define bsg_print_int_dbg(i)                    \
    bsg_print_int(i)
#else
#define bsg_print_int_dbg(i)
#endif

extern "C" int sparse_transpose(
    csr_matrix_t *I,
    csr_matrix_t *O,
    int *dram_buffer)
{
    appl::runtime_init(dram_buffer);
    appl::sync();
    bsg_cuda_print_stat_kernel_start();
    
    if (__bsg_id == 0) {
        int *O_nz_pos =  (int*)appl::appl_malloc(sizeof(int)*I->nnz);
        int *O_nnz = (int*)appl::appl_malloc(sizeof(int)*(I->n+1));
        appl::parallel_for(0, I->n+1, [=](int i){
                O_nnz[i] = 0;
            });
        // step 1. find non-zeros in each column
        // mark relative position of each non-zero
        bsg_print_hexadecimal(0xAAAAAAAA);
        appl::parallel_for(0, I->n, [=](int i){
                int start = I->rowptrs[i];
                int stop  = I->rowptrs[i+1];
                bsg_print_int_dbg(1000000 + i);
                bsg_print_int_dbg(2000000 + (stop-start));
                for (int nz = start; nz < stop; nz++) {
                    int j = I->nonzeros[nz].col;
                    bsg_print_int_dbg(3000000+j);
                    int loc = bsg_amoadd(
                        &O_nnz[j]
                        ,1
                        );
                    bsg_print_int_dbg(4000000+loc);
                    O_nz_pos[nz] = loc;
                }
            });
        bsg_print_hexadecimal(0xBBBBBBBB);
        // prefix sum
        parallel_prefix_sum(
            O_nnz
            ,O->rowptrs
            ,O->n
            );
        // transpose non-zeros
        bsg_print_hexadecimal(0xCCCCCCCC);
        appl::parallel_for(0, I->n, [=](int i){
                int start = I->rowptrs[i];
                int stop  = I->rowptrs[i+1];
                bsg_print_int_dbg(1000000+i);
                bsg_print_int_dbg(2000000+(stop-start));
                for (int nz = start; nz < stop; nz++) {
                    int j   = I->nonzeros[nz].col;
                    float v = I->nonzeros[nz].val;
                    bsg_print_int_dbg(3000000+j);
                    bsg_print_int_dbg(4000000+O->rowptrs[j]);
                    bsg_print_int_dbg(5000000+O_nz_pos[nz]);
                    O->nonzeros[O->rowptrs[j] + O_nz_pos[nz]]
                        = { i, v };
                }
            });
        // sort each row in the transpose
        bsg_print_hexadecimal(0xDDDDDDDD);
        appl::parallel_for(0, O->n, [=](int i){
                int start = O->rowptrs[i];
                int stop  = O->rowptrs[i+1];
                bsg_print_int_dbg(1000000 + i);
                bsg_print_int_dbg(2000000 + (stop-start));
                std::sort(O->nonzeros+start
                          ,O->nonzeros+stop
                          ,[](const csr_matrix_tuple_t &t0
                              ,const csr_matrix_tuple_t &t1) {
                              return t0.col < t1.col;
                          });
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
