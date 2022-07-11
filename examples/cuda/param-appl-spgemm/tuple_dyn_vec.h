#include "tuple_vec.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * used to keep a dynamically sized array of partials
 */
typedef struct tuple_dyn_vec {
    int   size;
    tuple_t *v;
} tuple_dyn_vec_t;

/**
 * initialize a partial buffer to hold at least max_size partials
 */
inline void tuple_dyn_vec_init(tuple_dyn_vec_t *dv, int max_size)
{
    if (max_size > 0) {
        dv->v = tuple_new(max_size);
    } else {
        dv->v = 0;
    }
    dv->size = 0;
}

/**
 * performs cleanup for a partial buffer
 */
inline void tuple_dyn_vec_exit(tuple_dyn_vec_t *dv)
{
    dv->size = 0;
    if (dv->v != 0) {
        tuple_free(dv->v);
        dv->v = 0;
    }
}

/**
 * move a partial buffer
 */
inline void tuple_dyn_vec_move(tuple_dyn_vec_t *to, tuple_dyn_vec_t *from)
{
    // clean up to
    tuple_dyn_vec_exit(to);

    // move to parameters
    to->size = from->size;
    to->v = from->v;

    // clear from
    from->size = 0;
    from->v = 0;
    tuple_dyn_vec_exit(from);
}

/**
 * merge two partial buffers
 */
void tuple_dyn_vec_merge(tuple_dyn_vec_t  *first
                         ,tuple_dyn_vec_t *second
                         ,tuple_dyn_vec_t *merged_o);
    
#ifdef __cplusplus
}
#endif
