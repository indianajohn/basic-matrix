#include "gaussian_elimination.hpp"
#include "lup_decomposition.hpp"
#include <iostream>

namespace basic_matrix {

namespace {
const double lupDeterminant(const Matrix &L, const Matrix &U,
                            const size_t &num_pivots) {
  double det = pow(-1.0, num_pivots);
  for (size_t x = 0; x < L.width(); x++) {
    det *= L(x, x);
  }
  for (size_t x = 0; x < U.width(); x++) {
    det *= U(x, x);
  }
  return det;
}
}; // namespace

size_t pivot(const basic_matrix::Matrix &A, basic_matrix::Matrix &P) {
  size_t n = 0;
  // Store the row index in the matrix.
  Matrix R(1, A.height());
  for (size_t y = 0; y < R.height(); y++) {
    R(0, y) = y;
  }

  // Matrix with a row "label" on the right.
  Matrix AR = A.concatRight(R);
  for (size_t x = 0; x < A.width(); x++) {
    // Find the max row index for a column.
    size_t y_for_max = 0;
    double max_y = fabs(AR(x, 0));
    for (size_t y = 1; y < AR.height(); y++) {
      double val = fabs(AR(x, y));
      if (val > max_y) {
        max_y = val;
        y_for_max = y;
      }
    }
    // Swap rows such that the max element in the column is on the diagonal.
    // (x, x) is on the diagonal, so we swap the max y with row y = x;
    AR.swapRows(x, y_for_max);
    n += (x != y_for_max);
  }
  // Form the permutation matrix based on the tracked row labels.
  P = Matrix(A.width(), A.height());
  // The row at which to find the row labels in AR - the last row of AR.
  size_t row_labels_x = AR.width() - 1;
  for (size_t y = 0; y < A.height(); y++) {
    size_t dst_row = AR(row_labels_x, y);
    // Setting the column index correspondong to the target row guarantees that
    // P * A will have the right elements in the row. Observe:
    // [0 1 0]   [1.1 1.2 1.3]   [2.1 2.2 2.3]
    // [1 0 0] * [2.1 2.2 2.3] = [1.1 1.2 1.3]
    // [0 0 1]   [3.1 3.2 3.3]   [3.1 3.2 3.3]
    P(dst_row, y) = 1.0;
  }
  return n;
}

bool lupDecomposition(const Matrix &A, Matrix &L, Matrix &U, Matrix &P) {
  size_t num_pivots = pivot(A, P);
  Matrix PA = P * A;
  if (!luDecomposition(PA, L, U)) {
    return false;
  }
  double det = lupDeterminant(L, U, num_pivots);
  if (fabs(det) < 1e-4) {
    return false;
  }
  return true;
}

bool luDecomposition(const Matrix &A, Matrix &L, Matrix &U) {
  if (A.width() != A.height()) {
    throw std::runtime_error(
        "Input matrix was " + std::to_string(A.width()) + "x" +
        std::to_string(A.height()) +
        " but a square matrix is required for lu decomposition.");
  }
  L = Matrix(A.width(), A.height());
  U = Matrix(A.width(), A.height());

  for (size_t i = 0; i < U.width(); i++) {
    U(i, i) = 1;
  }

  for (size_t j = 0; j < A.width(); j++) {
    for (size_t i = j; i < A.width(); i++) {
      double sum = 0.;
      for (size_t k = 0; k < j; k++) {
        sum += L(k, i) * U(j, k);
      }
      L(j, i) = A(j, i) - sum;
    }

    for (size_t i = j; i < A.width(); i++) {
      double sum = 0;
      for (size_t k = 0; k < j; k++) {
        sum = sum + L(k, j) * U(i, k);
      }
      if (L(j, j) == 0) {
        return false;
      }
      U(i, j) = (A(i, j) - sum) / L(j, j);
    }
  }
  return true;
}

void solveL(const Matrix &L, Matrix &b) {
  if (L.width() != L.height()) {
    throw std::runtime_error(std::to_string(L.width()) + "x" +
                             std::to_string(L.height()) +
                             " matrix is not square. solveL only works on a "
                             "square lower-triangular matrix.");
  }
  if (L.height() == 0) {
    return;
  }
  b(0, 0) = b(0, 0) / L(0, 0);
  for (size_t y = 1; y < L.height(); y++) {
    double b_result = b(0, y);
    for (size_t x = 0; x < y; x++) {
      b_result -= L(x, y) * b(0, x);
    }
    b(0, y) = b_result / L(y, y);
  }
}

void solveU(const Matrix &U, Matrix &b) {
  int start_y = std::min(U.height(), U.width()) - 1;
  b(0, start_y) = b(start_y, 0) / U(start_y, start_y);
  for (int y = start_y - 1; y >= 0; y--) {
    double b_result = b(0, y);
    for (int x = start_y; x > y; x--) {
      b_result -= U(x, y) * b(0, x);
    }
    b(0, y) = b_result / U(y, y);
  }
}

void solveLUP(const Matrix &L, const Matrix &U, const Matrix &P, Matrix &b) {
  b = P * b;
  solveL(L, b);
  solveU(U, b);
}

double lupDeterminant(const basic_matrix::Matrix &A) {
  if (A.width() == 0) {
    return 0.0;
  }
  Matrix P, L, U;
  size_t num_pivots = pivot(A, P);
  Matrix PA = P * A;
  luDecomposition(PA, L, U);
  return lupDeterminant(L, U, num_pivots);
}
}; // namespace basic_matrix
