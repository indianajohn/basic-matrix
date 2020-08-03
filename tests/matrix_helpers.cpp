#include "matrix_helpers.hpp"
#include "matrix.hpp"
#include <iostream>
#include <random>
#include <sstream>

using namespace basic_matrix;

bool matrixNear(const Matrix &mat_result, const Matrix &mat_expected,
                const double &tol) {
  bool near = true;
  if (mat_result.width() != mat_expected.width()) {
    std::cerr << "Matrices have different width" << std::endl;
    near = false;
  }
  if (mat_result.height() != mat_expected.height()) {
    std::cerr << "Matrices have different height " << std::endl;
    near = false;
  }
  std::stringstream stream;
  for (size_t x = 0; x < mat_result.width(); x++) {
    for (size_t y = 0; y < mat_result.height(); y++) {
      if (fabs(mat_result(x, y) - mat_expected(x, y)) > tol) {
        near = false;
        stream << "(" << x << "," << y << ")";
      }
    }
  }
  if (stream.str().size() > 0) {
    std::cerr << "Elements differ:" << stream.str() << std::endl;
  }

  if (!near) {
    return false;
  }
  return true;
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
