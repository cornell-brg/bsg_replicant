#include <algorithm>
#include "csr_matrix.h"
#include "appl.hpp"
#include "parallel_prefix_sum.hpp"

extern "C" int sparse_transpose(
    csr_matrix_t *I,
    csr_matrix_t *O,
    int *dram_buffer)
{
    appl::runtime_init(dram_buffer);
    appl::sync();
    bsg_cuda_print_stat_kernel_start();
    
    if (__bsg_id == 0) {
        int *O_nz_pos =  (int*)appl::malloc(sizeof(int)*I->nnz);
        int *O_nnz = (int*)appl::malloc(sizeof(int)*(I->n+1));
        appl::parallel_for(0, I->n+1, [=](int i){
                O_nnz[i] = 0;
            });
        // step 1. find non-zeros in each column
        // mark relative position of each non-zero
        appl::parallel_for(0, I->n, [=](int i){
                int start = I->rowptrs[i];
                int stop  = I->rowptrs[i+1];
                for (int nz = start; start < stop; nz++) {
                    int j = I->nonzeros[nz].col;
                    int loc = bsg_amoadd(
                        &O_nnz[j+1]
                        ,1
                        );
                    O_nz_pos[nz] = loc;
                }
            });
        // prefix sum
        parallel_prefix_sum(
            O_nnz
            ,O->rowptrs
            ,O->n
            );
        // transpose non-zeros
        appl::parallel_for(0, I->n, [=](int i){
                int start = I->rowptrs[i];
                int stop  = I->rowptrs[i+1];
                for (int nz = start; nz < stop; nz++) {
                    int j   = I->nonzeros[nz].col;
                    float v = I->nonzeros[nz].val;
                    O->nonzeros[O->rowptrs[j] + O_nz_pos[nz]]
                        = { i, v };
                }
            });
        // sort each row in the transpose
        appl::parallel_for(0, O->n, [=](int i){
                int start = O->rowptrs[i];
                int stop  = O->rowptrs[i+1];
                std::sort(O->nonzeros+start
                          ,O->nonzeros+stop
                          ,[](const csr_matrix_tuple_t &t0
                              ,const csr_matrix_tuple_t &t1) {
                              return t0.col < t1.cal;
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
