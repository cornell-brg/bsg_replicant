#pragma once
#include "mmio.h"
#include "coo_matrix.h"
#ifdef __cplusplus
extern "C" {
#endif
int  coo_matrix_init_from_mm(coo_matrix_t *mtx, FILE *mmfile);
#ifdef __cplusplus
}
#endif
