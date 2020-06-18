#include "gaussian_elimination.hpp"
#include "matrix.hpp"
#include "matrix_helpers.hpp"
#include "test_helpers.hpp"
#include <iostream>

using namespace basic_matrix;

void doesNotChangeIdentity() {
  Matrix mat_output({{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}});
  Matrix mat_expected({{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}});
  gaussianElimination(mat_output);
  assertMatrixNear(mat_expected, mat_output);
}

void linearlyDependentRows() {
  Matrix mat_output({{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}});
  Matrix mat_expected({{1, 1, 1, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}});
  gaussianElimination(mat_output);
  assertMatrixNear(mat_expected, mat_output);
}

void rankTwo() {
  Matrix mat_output({{1, 0, 0, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}});
  Matrix mat_expected({{1, 0, 0, 1}, {0, 1, 1, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}});
  gaussianElimination(mat_output);
  assertMatrixNear(mat_expected, mat_output);
}

void solveByGaussianEliminationWorks() {
  size_t num_trials = 20;
  for (size_t trial = 0; trial < num_trials; trial++) {
    size_t width = randomInt<size_t>(1, 8);
    size_t height = width;
    Matrix A = generateNonsingularMatrix(width, height);
    Matrix x = randomMatrix(1, A.height(), -10.0, 10.0);
    Matrix b = A*x;
    Matrix b_result = b;
    solveByGaussianElimination(A, b_result);
    assertMatrixNear(x, b_result);
  }
}

int main() {
  doesNotChangeIdentity();
  linearlyDependentRows();
  rankTwo();
  solveByGaussianEliminationWorks();
}
