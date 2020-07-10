#pragma once
#include "matrix.hpp"

namespace basic_matrix {
/// Calculate Eigenvalues through the QR algorithm.
/// returns an N x 1 matrix that contains eigenvalues.
/// Note: the input to this function must be a square matrix.
/// This function will not calculate complex eigenvalues. A
/// matrix that is not ok() will be returned in that case.
Matrix eigenvalues(const Matrix &A, const double &convergence_threshold = 1e-3,
                   const size_t &max_iterations = 1000);
}; // namespace basic_matrix
