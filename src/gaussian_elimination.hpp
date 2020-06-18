#pragma once
#include "matrix.hpp"

namespace basic_matrix {
/// Perform Gaussian elimination in place on A.
void gaussianElimination(basic_matrix::Matrix &mat);

/// Solve the system of equations A * x = b, where A is square and A.height() ==
/// b.height(),
/// by Gaussian elimination.
/// @in A - the A matrix.
/// @in/out b = the b matrix. Replaced by x during computation.
void solveByGaussianElimination(const basic_matrix::Matrix &A,
                                basic_matrix::Matrix &b);
}; // namespace basic_matrix
///
