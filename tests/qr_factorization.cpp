#include "matrix_helpers.hpp"
#include "qr_factorization.hpp"
#include "test_helpers.hpp"

using namespace basic_matrix;

void obeysDefinition() {
  size_t num_trials = 100;
  size_t num_successes = 0;
  for (size_t trial = 0; trial < num_trials; trial++) {
    size_t w = randomInt(1, 10);
    size_t h = randomInt(1, 10);
    Matrix A = randomMatrix(w, h, -100.0, 100.0);
    Matrix Q;
    Matrix R = A;
    qrFactorize(Q, R);
    Matrix error_matrix = (Q * R - A);
    double error = error_matrix.norm();
    if (error < 1e-1) {
      num_successes++;
    }
  }
  // We would expect this test to fail in cases where
  // A has linearly dependent columns.
  std::cout << "Succeeded " << num_successes << "/" << num_trials << " times."
            << std::endl;
  ASSERT(num_successes > 0);
}

void qrSolveWorksForSquareMatrices() {
  size_t num_trials = 50;
  for (size_t trial = 0; trial < num_trials; trial++) {
    size_t width = randomInt<size_t>(1, 10);
    size_t height = width;
    Matrix A = generateNonsingularMatrix(width, height);
    Matrix x = randomMatrix(1, A.height(), -10.0, 10.0);
    Matrix b = A * x;
    Matrix b_result = b;
    solveQR(A, b_result);
    assertMatrixNear(x, b_result, 1e-2);
  }
}

void qrSolveWorksForOverconstrainedSystems() {
  size_t num_trials = 50;
  for (size_t trial = 0; trial < num_trials; trial++) {
    size_t width = randomInt<size_t>(1, 10);
    size_t height = width + randomInt<size_t>(1, 10);
    Matrix A = randomMatrix(width, height, -100.0, 100.0);
    Matrix x = randomMatrix(1, A.width(), -10.0, 10.0);
    Matrix b = A * x;
    Matrix b_result = b;
    solveQR(A, b_result);
    Matrix x_result(MatrixROI(0, 0, 1, A.width(), &b_result));
    assertMatrixNear(x, x_result, 1e-2);
  }
}

void qrSolveWorksForUnderconstrainedSystems() {
  size_t num_trials = 50;
  for (size_t trial = 0; trial < num_trials; trial++) {
    size_t height = randomInt<size_t>(1, 10);
    size_t width = height + randomInt<size_t>(1, 10);
    Matrix A = randomMatrix(width, height, -100.0, 100.0);
    Matrix x = randomMatrix(1, A.width(), -10.0, 10.0);
    Matrix b = A * x;
    Matrix b_result = b;
    solveQR(A, b_result);
    // Don't assert anything, just make sure code works. We
    // don't have enough equations to solve for X.
  }
}

int main() {
  obeysDefinition();
  qrSolveWorksForSquareMatrices();
  qrSolveWorksForOverconstrainedSystems();
  qrSolveWorksForUnderconstrainedSystems();
}
