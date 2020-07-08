#pragma once
#include "matrix.hpp"

namespace basic_matrix {
/// Calculate Eigenvalues through the QR algorithm.
/// returns an N x 1 matrix that contains eigenvalues.
/// Note: the input to this function must be a square matrix.
Matrix eigenvalues(const Matrix &A, const double &convergence_threshold = 1e-3);
}; // namespace basic_matrix
