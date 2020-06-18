#include "gaussian_elimination.hpp"
#include <iostream>

namespace basic_matrix {

namespace {
void addRow(Matrix &mat, const size_t &y_from, const size_t &y_to) {
  for (size_t x = 0; x < mat.width(); x++) {
    mat(x, y_to) += mat(x, y_from);
  }
}

void multiplyRow(Matrix &mat, const size_t &y, const double &factor) {
  for (size_t x = 0; x < mat.width(); x++) {
    mat(x, y) *= factor;
  }
}

void eliminateAlongColumn(Matrix &mat, const size_t &x_target,
                          const size_t &y_target) {
  {
    // Make mat(x_target, y_target) 1.0
    double factor = 1.0 / mat(x_target, y_target);
    multiplyRow(mat, y_target, factor);
  }
  for (size_t y = 0; y < mat.height(); y++) {
    if (y == y_target || fabs(mat(x_target, y)) < 1e-9) {
      continue;
    }
    // Make x_target, y -1.0
    double factor = -1.0 / mat(x_target, y);
    multiplyRow(mat, y, factor);
    // Add the target (now 1.0) row to this one (not -1.0),
    // zeroing this element out.
    addRow(mat, y_target, y);
    multiplyRow(mat, y, 1. / factor);
  }
}
} // namespace

void gaussianElimination(Matrix &mat) {
  for (size_t focus_x = 0; focus_x < mat.width(); focus_x++) {
    size_t focus_y = focus_x;
    for (size_t y = focus_y; y < mat.height(); y++) {
      if (fabs(mat(focus_x, y)) > 1e-9) {
        if (y != focus_y) {
          mat.swapRows(focus_y, y);
        }
        eliminateAlongColumn(mat, focus_x, focus_y);
        break;
      }
    }
  }
}

void solveByGaussianElimination(const basic_matrix::Matrix &A,
                                basic_matrix::Matrix &b) {
  Matrix A_copy = A;
  Matrix Ab_wrapped;
  Ab_wrapped.addROI(MatrixROI(0, 0, A.width(), A.height(), &A_copy));
  Ab_wrapped.addROI(MatrixROI(0, 0, 1, A.height(), &b, A.width(), 0));
  // A and b will be mutated in place, A is temporary, b was passed in.
  gaussianElimination(Ab_wrapped);
}
}; // namespace basic_matrix
