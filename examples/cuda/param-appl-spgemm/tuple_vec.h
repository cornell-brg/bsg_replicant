#pragma once
#include <limits.h>
#include "csr_matrix.h"
#include "list.h"
#ifdef __cplusplus
extern "C" {
#endif

    /**
     * shorthand for tuple_t
     */
    typedef csr_matrix_tuple_t tuple_t;

    /**
     *  a tuple vec
     */
    typedef struct tuple_vec {
        int           size;
        list_node_t   free;
        tuple_t       v[1];
    }   tuple_vec_t;

    /**
     * return a tuple_vec header from a list node
     */
    inline tuple_vec_t *tuple_vec_from_list_node(list_node_t *node)
    {
        return (tuple_vec_t*)(
            ((char*)node)
            - offsetof(tuple_vec_t, free)
            );
    }

    /**
     * return a tuple_vec header from the first partial in a buffer
     */
    inline tuple_vec_t *tuple_vec_from_tuple(tuple_t *tp)
    {
        return (tuple_vec_t*)(
            ((char *)(tp))
            - offsetof(tuple_vec_t, v)
            );
    }

    /**
     * lists of free vectors
     */
    enum {
        TUPLE_VEC_FREE_SMALL
        ,TUPLE_VEC_FREE_MEDIUM
        ,TUPLE_VEC_FREE_LARGE
        ,TUPLE_VEC_FREE_HUGE
        ,TUPLE_VEC_FREE_N
    };    
    extern list_t tuple_vec_free [TUPLE_VEC_FREE_N];

   /**
    * maps a size to a bucket with the smallest buffers
    * that can fit size
    */
    inline int tuple_vec_bucket(int size)
    {
        // decide bucket 
        if (size <= 128) {
            return TUPLE_VEC_FREE_SMALL;
        } else if (size <= 1024) {
            return TUPLE_VEC_FREE_MEDIUM;
        } else if (size <= 8192){
            return TUPLE_VEC_FREE_LARGE;
        } else {
            return TUPLE_VEC_FREE_N;
        }
    }

   /**
    * maps a bucket to a size
    */
    inline int tuple_vec_bucket_to_size(int bkt) {
        switch (bkt) {
        case TUPLE_VEC_FREE_SMALL:  return     128;
        case TUPLE_VEC_FREE_MEDIUM: return    1024;
        case TUPLE_VEC_FREE_LARGE:  return    8192;
        case TUPLE_VEC_FREE_HUGE:   return INT_MAX;
        default:                    return       0;
        }
    }

   /**
    * allocate a new buffer of partials large enough to fit 'size'
    * partials
    */
    tuple_t    *tuple_new(int size);
    inline void tuple_free(tuple_t *tp) {
        tuple_vec_t *tv = tuple_vec_from_tuple(tp);
        int bkt = tuple_vec_bucket(tv->size);
        list_append(&tuple_vec_free[bkt], &tv->free);
    }

   /**
    * initialize the lib
    */    
    void tuple_vec_libinit();

#ifdef __cplusplus
}
#endif
