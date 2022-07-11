#pragma once
#include <stdlib.h>
#include "bsg_manycore.h"
#include "bsg_manycore_cuda.h"
#include "bsg_manycore_regression.h"
#ifdef __cplusplus
extern "C" {
#endif

static inline int try_malloc(size_t sz, void **ptr)
{
    void *p = malloc(sz);
    if (p == NULL) {
        return HB_MC_NOMEM;
    }
    *ptr = p;
    return HB_MC_SUCCESS;
}
#ifdef __cplusplus
}
#endif
