#pragma once
#include "Eigen/Core"
#include "Eigen/Sparse"
#include <vector>
#include <iostream>
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

// copied from sparse matrix motif benchmark
namespace eigen_sparse_matrix {
    /* Compare non-zeros along major axis and return true if they are approximately equal */
    template <class SparseMatrixType>
    bool mjr_equal(SparseMatrixType &A, SparseMatrixType &B, typename SparseMatrixType::StorageIndex mjr) {
        using Index = typename SparseMatrixType::StorageIndex;
        using Scalar = typename SparseMatrixType::Scalar;
        Index nnz_A = A.innerNonZeroPtr()[mjr];
        Index nnz_B = B.innerNonZeroPtr()[mjr];
#ifdef DEBUG_EIGEN_SPARSE_MATRIX
        std::cout << "nnz(A," << mjr << ")=" << nnz_A << std::endl;
        std::cout << "nnz(B," << mjr << ")=" << nnz_B << std::endl;
#endif
        if (nnz_A != nnz_B) {
            return false;
        }

        Index nnz = nnz_A;
        // offsets
        Index off_A = A.outerIndexPtr()[mjr];
        Index off_B = B.outerIndexPtr()[mjr];
        // indices
        Index *idx_A = &A.innerIndexPtr()[off_A];
        Index *idx_B = &B.innerIndexPtr()[off_B];
        // values
        Scalar *val_A = &A.valuePtr()[off_A];
        Scalar *val_B = &B.valuePtr()[off_B];
        // for each non-zero
        for (Index nz = 0; nz < nnz; nz++) {
            // fetch indices
            Index iA = idx_A[nz];
            Index iB = idx_B[nz];
#ifdef DEBUG_EIGEN_SPARSE_MATRIX
            std::cout << "idx(A, " << nz << ")=" << iA << std::endl;
            std::cout << "idx(B, " << nz << ")=" << iB << std::endl;
#endif
            // fetch scalar
            Scalar vA = val_A[nz];
            Scalar vB = val_B[nz];
#ifdef DEBUG_EIGEN_SPARSE_MATRIX
            std::cout << "val(A, " << nz << ")=" << vA << std::endl;
            std::cout << "val(B, " << nz << ")=" << vB << std::endl;
#endif
            if (iA != iB)
                return false;

            if ((vA-vB) > std::numeric_limits<float>::epsilon())
                return false;
        }
        return true;
    }

    /* Compare non-zeros along major axis for range of major indices */
    template <class SparseMatrixType>
    bool mjr_range_equal(SparseMatrixType &A
                         , SparseMatrixType &B
                         , typename SparseMatrixType::StorageIndex mjr_lo
                         , typename SparseMatrixType::StorageIndex mjr_hi)
    {
        using Index = typename SparseMatrixType::StorageIndex;
        //using Scalar = typename SparseMatrixType::Scalar;
        if (A.isCompressed())
            A.uncompress();
        if (B.isCompressed())
            B.uncompress();
        for (Index mjr = mjr_lo; mjr < mjr_hi; mjr++) {
#ifdef DEBUG_EIGEN_SPARSE_MATRIX
            std::cout << "comparing row " << mjr << std::endl;
#endif
            if (!mjr_equal(A, B, mjr))
                return false;
        }
        return true;
    }
}

inline bool eigen_sparse_matrices_are_equal(
    eigen_sparse_matrix_t  & A
    ,eigen_sparse_matrix_t & B
    ) {
    return eigen_sparse_matrix::mjr_range_equal(A, B, 0, A.rows());
}

