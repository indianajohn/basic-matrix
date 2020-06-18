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
  Matrix L, U, P;
  if (!lupDecomposition(*this, L, U, P)) {
    // Return a not ok() matrix - - lup decomposition failed, meaning
    // the matrix was singular.
    return Matrix();
  }
  for (size_t x = 0; x < width(); x++) {
    Matrix e(1, height());
    e(0, x) = 1.0;
    solveLUP(L, U, P, e);
    for (size_t y = 0; y < height(); y++) {
      result(x, y) = e(0, y);
    }
  }
  return result;
}
