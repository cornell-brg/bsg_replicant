#include "appl.hpp"
#include "bsg_manycore.h"
#include "bsg_manycore_atomic.h"

inline int floor_log2(int x)
{
    int i = -1;
    int j = i+1;
    while  ((1 << j) <= x) {
        i = j;
        j = j+1;
    }
    return i;
}

inline int ceil_log2(int x)
{
    int j = 0;
    while  ((1 << j) <= x) {
        j = j+1;
    }
    return j;
}

inline int tree_rchild(int root)
{
    return 2*root + 2;
}

inline int tree_lchild(int root)
{
    return 2*root + 1;
}

inline void parallel_prefix_sum(
    int  *in
    ,int *out
    ,int  n
    ) {
    int *tree = (int*)appl::appl_malloc(sizeof(int)*2*appl::get_nthreads());

    appl::parallel_for(0, (int)appl::get_nthreads(), 1, [=](int tid){
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
    
    appl::parallel_for(0, (int)appl::get_nthreads(), [=](int tid){
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

