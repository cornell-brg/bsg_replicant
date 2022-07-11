#pragma once
#include "coo_matrix.h"
#include "csr_matrix.h"
#ifdef __cplusplus
extern "C" {
#endif
int  coo_matrix_init_from_csr(coo_matrix_t *coo, csr_matrix_t *csr);
#ifdef __cplusplus
}
#endif
