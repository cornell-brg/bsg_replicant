#include "bsg_manycore.h"
#include "tuple_dyn_vec.h"

template <int UNROLL>
static inline void tuples_copy(
    tuple_t        *__restrict__ dst
    ,const tuple_t *__restrict__ src
    ,int n
    )
{
    int i = 0;
    for (; i + UNROLL < n; i += UNROLL) {
        int   tx [UNROLL];
        float tv [UNROLL];
        bsg_unroll(8)
        for (int pre = 0; pre < UNROLL; pre++) {
            tx[pre] = src[i+pre].col;
            tv[pre] = src[i+pre].val;
        }
        bsg_compiler_memory_barrier();
        bsg_unroll(8)
        for (int pre = 0; pre < UNROLL; pre++) {
            dst[i+pre].col = tx[pre];
            dst[i+pre].val = tv[pre];
        }
    }

    for (; i < n; i++) {
        int   tx = src[i].col;
        float tv = src[i].val;
        dst[i].col = tx;
        dst[i].val = tv;
    }
}

/**
 * merge two partial buffers
 */
void tuple_dyn_vec_merge(
    tuple_dyn_vec_t  *first
    ,tuple_dyn_vec_t *second
    ,tuple_dyn_vec_t *merged_o
    ) {
    
    tuple_dyn_vec_t merged;
    tuple_dyn_vec_init(&merged, first->size + second->size);

    // bsg_print_hexadecimal((unsigned)first->v);
    // bsg_print_int(6000000 + first->size);
    // bsg_print_hexadecimal((unsigned)second->v);
    // bsg_print_int(6000000 + second->size);
    // bsg_print_hexadecimal((unsigned)merged.v);
    // bsg_print_int(6000000 + merged.size);
    
    int n0 = first->size, n1 = second->size;
    int i0 = 0, i1 = 0, im = 0;
    int x0, x1, x0_n, x1_n;
    float v0, v1, v0_n, v1_n;

    if (i0 < n0) {
        x0 = first->v[i0].col;
        v0 = first->v[i0].val;
        x0_n = first->v[i0+1].col;
        v0_n = first->v[i0+1].val;
    }
    if (i1 < n1) {
        x1 = second->v[i1].col;
        v1 = second->v[i1].val;
        x1_n = second->v[i1+1].col;
        v1_n = second->v[i1+1].val;
    }
    bsg_compiler_memory_barrier();
    while (i0 < n0 && i1 < n1) {
        int   xm;
        float vm;
        if (x0 < x1) {
            xm = x0;
            vm = v0;
            x0 = x0_n;
            v0 = v0_n;
            i0++;
            x0_n = first->v[i0+1].col;
            v0_n = first->v[i0+1].val;
        } else if (x1 < x0) {
            xm = x1;
            vm = v1;
            x1 = x1_n;
            v1 = v1_n;
            i1++;
            x1_n = second->v[i1+1].col;
            v1_n = second->v[i1+1].val;
        } else {
            xm = x0;
            vm = v0+v1;
            x0 = x0_n;
            v0 = v0_n;
            x1 = x1_n;
            v1 = v1_n;
            i0++;
            i1++;
            x0_n = first->v[i0+1].col;
            v0_n = first->v[i0+1].val;
            x1_n = second->v[i1+1].col;
            v1_n = second->v[i1+1].val;

        }
        bsg_compiler_memory_barrier();
        // write back
        merged.v[im].col = xm;
        merged.v[im].val = vm;
        im++;
    }
    // copy which ever buffer remains
    tuples_copy<4>(&merged.v[im], &first->v[i0],  n0-i0);
    tuples_copy<4>(&merged.v[im], &second->v[i1], n1-i1);

    // set the size, add remainder
    merged.size = im + (n0-i0) + (n1-i1);

    // clear first + second
    tuple_dyn_vec_exit(first);
    tuple_dyn_vec_exit(second);

    // move merged -> merged_o
    tuple_dyn_vec_move(merged_o, &merged);
}

