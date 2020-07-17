#pragma once
#include "matrix.hpp"

namespace basic_matrix {

/// Perform QR factorization on the matrix R. Save the result in
/// R.
///
/// A QR factorization decomposes the input matrix A into Q an R
/// such that:
/// A  = Q * R
///
/// Where Q is an orthogonal matrix and R is an upper-triangular
/// matrix. One application of this algorithm is the computation
/// of matrix eigenvalues.
///
/// The Householder method is used here. The method is described
/// here: http://www.seas.ucla.edu/~vandenbe/ee133a.html
/// Complexity is 2*height*width^2 - (2/3)*n^3 flops
///
/// out: Q
/// in/out: R
void qrFactorize(Matrix &Q, Matrix &R);

/// Solve a system of linear equations using QR factorization.
/// This system does not have to be square. Common applications
/// of this method are linear least squares problems. This method
/// is capable of dealing with square matrices, underconstrained
/// problems, and overconstrained problems.
///
/// This implementation uses the Householder method, widely known
/// to be the most numerically stable. The calculation takes
/// 2 * height*width^2 flops
///
/// in: A
/// in/out: b, x is saved here at the end.
void solveQR(const Matrix &A, Matrix &b);
}; // namespace basic_matrix
