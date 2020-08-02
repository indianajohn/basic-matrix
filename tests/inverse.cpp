#include "matrix.hpp"
#include "matrix_helpers.hpp"
#include "test_helpers.hpp"
#include <iostream>

using namespace basic_matrix;

void nonSingularMatricesWork() {
  size_t trial_count = 20;
  for (size_t i = 0; i < trial_count; i++) {
    size_t width = randomInt(1, 10);
    size_t height = width;
    Matrix A = randomMatrix(width, height, -100.0, 100.0);
    Matrix AI = A.inverse();
    if (!AI.ok()) {
      std::cerr << "could not invert invertible matrix: " << A << std::endl;
      ASSERT(false);
    }
    // By definition, multiplying a matrix by its inverse results in an
    // identity matrix.
    Matrix AAI = A * AI;
    Matrix I = identity(width);
    ASSERT_MATRIX_NEAR(AAI, I);
  }
}

int main() { nonSingularMatricesWork(); }
