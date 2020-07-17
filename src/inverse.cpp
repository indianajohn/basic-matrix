#include "gaussian_elimination.hpp"
#include "lup_decomposition.hpp"
#include "matrix.hpp"

using namespace basic_matrix;

Matrix Matrix::inverse() const {
  if (width() != height()) {
    throw std::runtime_error(
        "Inverse only makes sense for square matrices; this matrix is " +
        std::to_string(width()) + "x" + std::to_string(height()));
  }
  Matrix result(width(), height());
  // Calculate an LUP decomposition so that we can solve the below
  // systems with a lower computational cost. This has the same effect
  // as solving these systems through other means, such as Gaussian
  // elimination.
  Matrix L, U, P;
  if (!lupDecomposition(*this, L, U, P)) {
    // Return a not ok() matrix - lup decomposition failed, meaning
    // the matrix was singular.
    return Matrix();
  }
  // Computing the inverse is equivalent to the following calculation:
  // A * A.inverse() = I
  // e.g.
  // [a11 a12] * [ai11 ai12] = [1 0]
  // [a21 a22]   [ai21 ai22]   [0 1]
  //
  // This is equivalent to solving N linear systems, where N is the
  // number of rows/columns.
  // One system:
  // a11 * ai11 + a12 * ai21 = 1
  // a21 * ai11 + a22 * ai21 = 0
  for (size_t x = 0; x < width(); x++) {
    // Create a system equivalent to
    // A * x = zeros, with a single 1 on the column. For example,
    // [1 2] * [a11i] = [1]
    // [3 4] * [a12i]   [0]
    Matrix e(1, height());
    e(0, x) = 1.0;
    solveLUP(L, U, P, e);
    // Store the result in the appropriate row.
    for (size_t y = 0; y < height(); y++) {
      result(x, y) = e(0, y);
    }
  }
  return result;
}
