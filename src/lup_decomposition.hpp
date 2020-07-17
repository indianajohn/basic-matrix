#pragma once
#include "matrix.hpp"

namespace basic_matrix {
///
/// Perform LUP decomposition of A. The result satisfies:
/// PA = LU
/// Where P is a permutation matrix, L is a lower-diagonal matrix, and
/// U is an upper-diagonal matrix.
/// LUP decompositions have a couple of applications.
///
/// 1. Create a "cache" whereby linear systems involving the matrix A
///    can be solved with fewer FLOPs. this is useful when, for example,
///    computing the matrix inverse, which involves solving different
///    linear systems A*x = b_i for the i rows of matrix A.
///
/// 2. Computing the determinant in a computationally-efficient manner.
bool lupDecomposition(const basic_matrix::Matrix &A, basic_matrix::Matrix &L,
                      basic_matrix::Matrix &U, basic_matrix::Matrix &P);

/// Pivot matrix : Determine P such that P*A results in a large diagonal.
size_t pivot(const basic_matrix::Matrix &A, basic_matrix::Matrix &P);

/// Perform LU decomposition of A.
bool luDecomposition(const basic_matrix::Matrix &A, basic_matrix::Matrix &L,
                     basic_matrix::Matrix &U);

/// Compute the determinant using LUP decomposition.
double lupDeterminant(const basic_matrix::Matrix &A);

/// Solve the linear system L*x = b, where L is a lower-diagonal matrix,  using
/// back substitution. This will produce the same result to Gaussian elimination
/// with less computation.
/// @in L - a lower triangular matrix.
/// @in/out b - b in L*x = b. x is stored in b as the result.
void solveL(const Matrix &L, Matrix &b);

/// Solve a linear system U*x = b, where U is an upper-diagonal matrix, using
/// back substitution. Again, this produces the same result as Gaussian
/// elimination with less computation.
/// @in U - an upper triangular matrix.
/// @in/out b - b in L*x = b. x is stored in b as the result.
void solveU(const Matrix &L, Matrix &b);

/// Solve the system L*U*x = P*b. In this system, x is the same as the system
/// A*x = b, for the A that created the L, U, and P matrix via LUP
/// decomposition.
/// @in L, U, P -- result of lupDecomposition.
/// @in/out b - right-hand side of equation. Output is stored here.
void solveLUP(const Matrix &L, const Matrix &U, const Matrix &P, Matrix &b);

}; // namespace basic_matrix
