#define DEBUG
#include <algorithm>
#include <cstring>
#include "bsg_manycore_cuda.h"
#include "bsg_manycore_printing.h"
#include "utils.h"
#include "csr_matrix.h"

int csr_matrix_init_from_coo(csr_matrix_t *csr, coo_matrix_t *coo)
{
    csr_matrix_tuple_t *tuples;
    int *rowptrs;
    int *rowcnt;

    csr->n = coo->n;
    csr->nnz = coo->nz;
    printf("Initializing CSR for %dx%d Matrix with %d NZ\n"
           , coo->n
           , coo->n
           , coo->nz);
    bsg_pr_dbg("%s: allocating output buffers\n", __func__);
    // allocate output buffers
    BSG_CUDA_CALL(try_malloc(sizeof(csr_matrix_tuple_t)*(csr->nnz), (void**)&tuples));
    BSG_CUDA_CALL(try_malloc(sizeof(int)*(csr->n+1), (void**)&rowptrs));

    rowptrs[csr->n] = csr->nnz;

    // allocate temporary row count buffer
    bsg_pr_dbg("%s: allocating temporary count buffers\n", __func__);
    BSG_CUDA_CALL(try_malloc(sizeof(int)*csr->n, (void**)&rowcnt));
    memset(rowcnt, 0, sizeof(int)*csr->n);
    for (int nz = 0; nz < csr->nnz; nz++) {
        rowcnt[coo->nonzeros[nz].row]++;
    }

    // perform scan
    bsg_pr_dbg("%s: performing degree scan\n", __func__);
    int cum = 0;
    for (int row = 0; row < csr->n; row++) {
        rowptrs[row] = cum;
        cum += rowcnt[row];
    }

    // copy non-zeros one-by-one
    bsg_pr_dbg("%s: copying nonzeros\n", __func__);
    memset(rowcnt, 0, sizeof(int)*csr->n);
    for (int nz = 0; nz < csr->nnz; nz++) {
        coo_matrix_tuple_t *tuple = &coo->nonzeros[nz];
        bsg_pr_dbg("%s: copying nz %d: row %d, col %d, val %f\n"
                   , __func__
                   , nz
                   , tuple->row
                   , tuple->col
                   , tuple->val);
        bsg_pr_dbg("%s: copying nz %d: rowptrs[%d] = %d, rowcnt[%d] = %d\n"
                   , __func__
                   , nz
                   , tuple->row
                   , rowptrs[tuple->row]
                   , tuple->row
                   , rowcnt[tuple->row]);
        
        tuples[rowptrs[tuple->row]+rowcnt[tuple->row]].col = tuple->col;
        tuples[rowptrs[tuple->row]+rowcnt[tuple->row]].val = tuple->val;
        rowcnt[tuple->row]++;
    }

    // sort each row by column
    bsg_pr_dbg("%s: sort each row by column\n", __func__);
    for (int row = 0; row < csr->n; row++) {
        int degree = rowptrs[row+1]-rowptrs[row];
        csr_matrix_tuple_t *start = &tuples[rowptrs[row]];
        csr_matrix_tuple_t *stop  = start + degree;
        std::sort(start, stop, [](const csr_matrix_tuple_t &lhs, const csr_matrix_tuple_t &rhs){
                return lhs.col < rhs.col;
            });
    }

    // set members
    csr->rowptrs = rowptrs;
    csr->nonzeros = tuples;

    // free temporary memory
    free(rowcnt);

    return HB_MC_SUCCESS;
}

void csr_matrix_dest(csr_matrix_t *csr)
{
    csr->n = 0;
    csr->nnz = 0;
    free(csr->rowptrs);
    free(csr->nonzeros);
    csr->rowptrs = NULL;
    csr->nonzeros = NULL;
    return;
}

