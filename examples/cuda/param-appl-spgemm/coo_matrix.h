#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef struct coo_matrix_tuple {
    int   row;
    int   col;
    float val;
} coo_matrix_tuple_t;

typedef struct coo_matrix {
    int   n;
    int   nz;
    coo_matrix_tuple_t *nonzeros; 
} coo_matrix_t;

#ifdef __cplusplus
}
#endif
