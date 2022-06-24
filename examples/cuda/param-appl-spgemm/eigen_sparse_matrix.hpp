#pragma once
#include "Eigen/Core"
#include "Eigen/Sparse"
#include <vector>
#include "coo_matrix.h"
#include "bsg_manycore_printing.h"

typedef Eigen::SparseMatrix<float, Eigen::RowMajor, int> eigen_sparse_matrix_t;
typedef Eigen::Triplet<float> eigen_sparse_triplet_t;

static inline eigen_sparse_matrix_t eigen_sparse_matrix_from_coo(coo_matrix_t *coo) {
    eigen_sparse_matrix_t matrix (coo->n, coo->n);
    std::vector<eigen_sparse_triplet_t> triplets;
    for (int nz = 0; nz < coo->nz; nz++) {
        coo_matrix_tuple_t *tuple = &coo->nonzeros[nz];
        triplets.push_back(eigen_sparse_triplet_t(tuple->row, tuple->col, tuple->val));
    }
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    return matrix;
}
