#include "tuple_vec.h"
#include "appl.hpp"

list_t tuple_vec_free [TUPLE_VEC_FREE_N];

void tuple_vec_libinit()
{
    for (int i = 0; i < TUPLE_VEC_FREE_N; i++) {
        list_init(&tuple_vec_free[i]);
    }
}

/**
 * allocate a new buffer of partials large enough to fit 'size'
 * partials
 */
tuple_t *tuple_new(int size)
{
    int bkt = tuple_vec_bucket(size);
    tuple_vec_t *tv;
    if (!list_empty(&tuple_vec_free[bkt])) {
        list_node_t *n = list_front(&tuple_vec_free[bkt]);
        list_pop_front(&tuple_vec_free[bkt]);
        return &tuple_vec_from_list_node(n)->v[0];
    } else {
        // allocate enough for the max size of this bucket
        size = tuple_vec_bucket_to_size(bkt);
        int alloc_size = size * sizeof(tuple_t);
        
        // include extra for the buffer header
        alloc_size += sizeof(tuple_vec_t);
        
        // allocate
        tv = (tuple_vec_t*)appl::appl_malloc(alloc_size);
        // bsg_print_int(7000000 + size);
        // bsg_print_hexadecimal((unsigned)tv);

        tv->size = tuple_vec_bucket_to_size(bkt);
        tv->free.next = nullptr;

        return &tv->v[0];
    }
}
