#include "matrix_helpers.hpp"
#include "matrix.hpp"
#include <iostream>
#include <random>

using namespace basic_matrix;

void assertMatrixNear(const Matrix &mat_result, const Matrix &mat_expected,
                      const double &tol) {
  ASSERT_EQ(mat_result.width(), mat_expected.width());
  ASSERT_EQ(mat_result.height(), mat_expected.height());
  bool near = true;
  for (size_t x = 0; x < mat_result.width(); x++) {
    for (size_t y = 0; y < mat_result.height(); y++) {
      if (fabs(mat_result(x, y) - mat_expected(x, y)) > tol) {
        near = false;
      }
    }
  }
  if (!near) {
    std::cerr << "The matrix:" << std::endl
              << mat_result
              << " was supposed to be near the matrix: " << std::endl
              << mat_expected << "but wasn't.";
    ASSERT(false);
  }
}

double randomDouble(const double &min, const double &max) {
  std::uniform_real_distribution<double> dist(min, max);
  return dist(gen);
}

Matrix randomMatrix(const size_t &width, const size_t &height,
                    const double &min, const double &max,
                    const double &min_absolute_value) {
  Matrix result(width, height);
  for (size_t x = 0; x < result.width(); x++) {
    for (size_t y = 0; y < result.height(); y++) {
      double val = randomDouble(min, max);
      while (fabs(val) < min_absolute_value) {
        val = randomDouble(min, max);
      }
      result(x, y) = val;
    }
  }
  return result;
}

Matrix generateNonsingularMatrix(const size_t &width, const size_t &height) {
  Matrix A = randomMatrix(width, height, -100.0, 100.0);
  while (A.det() < 1e-3) {
    A = randomMatrix(width, height, -100.0, 100.0);
  }
  return A;
}

void assertContains(const basic_matrix::Matrix &mat_result,
                    const double &target, const double &tol) {
  for (size_t u = 0; u < mat_result.width(); u++) {
    for (size_t v = 0; v < mat_result.height(); v++) {
      if (fabs(mat_result(u, v) - target) < tol) {
        return;
      }
    }
  }
  throw std::runtime_error("Matrix did not contain value: " +
                           std::to_string(target));
}
