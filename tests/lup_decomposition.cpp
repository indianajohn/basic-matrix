#include "lup_decomposition.hpp"
#include "gaussian_elimination.hpp"
#include "matrix.hpp"
#include "matrix_helpers.hpp"
#include "test_helpers.hpp"
#include <iostream>

using namespace basic_matrix;

Matrix createLowerTriangularMatrix() {
  size_t width = randomInt<size_t>(1, 20);
  size_t height = width;
  // Generate a random matrix without zeros.
  Matrix result = randomMatrix(width, height, -10.0, 10.0);
  // Zero out the upper-triangular region.
  for (size_t y = 0; y < result.height(); y++) {
    for (size_t x = y + 1; x < result.width(); x++) {
      result(x, y) = 0;
    }
  }
  return result;
}

Matrix createUpperTriangularMatrix() {
  size_t width = randomInt<size_t>(1, 20);
  size_t height = width;
  // Generate a random matrix without zeros.
  Matrix result = randomMatrix(width, height, -10.0, 10.0);
  // Zero out the upper-triangular region.
  for (size_t x = 0; x < result.width(); x++) {
    for (size_t y = x + 1; y < result.height(); y++) {
      result(x, y) = 0;
    }
  }
  return result;
}

void pivotWorks() {
  Matrix A = {{0, 3, 0}, {4, 0, 0}, {0, 0, 5}};
  Matrix PA_expected = {{4, 0, 0}, {0, 3, 0}, {0, 0, 5}};
  Matrix P;
  pivot(A, P);
  Matrix PA = P * A;
  ASSERT_MATRIX_NEAR(PA, PA_expected);
}

void luDecompositionObeysDefinition() {
  size_t trial_count = 20;
  size_t successCount = 0;
  for (size_t i = 0; i < trial_count; i++) {
    size_t width = randomInt(1, 20);
    size_t height = width;
    Matrix A = randomMatrix(width, height, -100.0, 100.0);
    Matrix L, U;
    bool result = luDecomposition(A, L, U);
    if (result) {
      successCount++;
      Matrix LU = L * U;
      ASSERT_MATRIX_NEAR_TOL(A, LU, 1e-2);
    }
  }
  ASSERT(successCount > 0);
  std::cout << "Success count: " << successCount << std::endl;
}

void lupDecompositionObeysDefinition() {
  size_t trial_count = 20;
  size_t successCount = 0;
  for (size_t i = 0; i < trial_count; i++) {
    size_t width = randomInt(1, 20);
    size_t height = width;
    Matrix A = randomMatrix(width, height, -100.0, 100.0);
    Matrix L, U, P;
    bool result = lupDecomposition(A, L, U, P);
    if (result) {
      successCount++;
      Matrix LU = L * U;
      Matrix PA = P * A;
      ASSERT_MATRIX_NEAR_TOL(PA, LU, 1e-2);
    }
  }
  ASSERT(successCount > 0);
  std::cout << "Success count: " << successCount << std::endl;
}

void lupDecompositionFailsForSingularMatrices() {
  size_t trial_count = 20;
  for (size_t i = 0; i < trial_count; i++) {
    size_t width = randomInt(5, 6);
    size_t height = width;
    Matrix A = randomMatrix(width, height, -100.0, 100.0);
    size_t row1 = randomInt<size_t>(0, width - 1);
    size_t row2 = randomInt<size_t>(0, width - 1);
    while (row2 == row1) {
      row2 = randomInt<size_t>(0, width - 1);
    }
    for (size_t x = 0; x < A.width(); x++) {
      A(x, row1) = 1.0;
      A(x, row2) = 1.0;
    }
    Matrix L, U, P;
    bool result = lupDecomposition(A, L, U, P);
    ASSERT(!result);
  }
}

void solveLWorks() {
  size_t num_trials = 20;
  for (size_t trial = 0; trial < num_trials; trial++) {
    Matrix L = createLowerTriangularMatrix();
    Matrix x = randomMatrix(1, L.height(), -10.0, 10.0);
    Matrix b = L * x;
    Matrix b_result = b;
    solveL(L, b_result);
    ASSERT_MATRIX_NEAR_TOL(x, b_result, 1e-2);
  }
}

void solveUWorks() {
  size_t num_trials = 1;
  for (size_t trial = 0; trial < num_trials; trial++) {
    Matrix U = createUpperTriangularMatrix();
    Matrix x = randomMatrix(1, U.height(), -10.0, 10.0);
    Matrix b = U * x;
    Matrix b_result = b;
    solveU(U, b_result);
    ASSERT_MATRIX_NEAR_TOL(x, b_result, 1e-2);
  }
}

void solveLUPWorks() {
  size_t num_trials = 20;
  for (size_t trial = 0; trial < num_trials; trial++) {
    size_t width = randomInt<size_t>(1, 20);
    size_t height = width;
    Matrix A = generateNonsingularMatrix(width, height);
    Matrix x = randomMatrix(1, A.height(), -10.0, 10.0);
    Matrix b = A * x;
    Matrix b_result = b;
    Matrix L, U, P;
    lupDecomposition(A, L, U, P);
    solveLUP(L, U, P, b_result);
    ASSERT_MATRIX_NEAR_TOL(x, b_result, 1e-2);
  }
}
void lupDeterminantWorks() {
  size_t num_trials = 20;
  for (size_t trial = 0; trial < num_trials; trial++) {
    // BF determinant is very slow above 5
    size_t width = randomInt<size_t>(1, 5);
    size_t height = width;
    Matrix A = generateNonsingularMatrix(width, height);
    double lup_det = lupDeterminant(A);
    double det = bruteForceDeterminant(A);
    ASSERT_TOL(det, lup_det, det * 0.00001);
  }
}

int main() {
  pivotWorks();
  luDecompositionObeysDefinition();
  lupDecompositionObeysDefinition();
  solveLWorks();
  solveUWorks();
  solveLUPWorks();
  lupDeterminantWorks();
  lupDecompositionFailsForSingularMatrices();
}
