#include "matrix.hpp"
#include <iostream>
#include <math.h>
#include <random>
#include <stdexcept>

namespace basic_matrix {
Matrix Matrix::transpose() const {
  Matrix result(height(), width());
  for (size_t x = 0; x < width(); x++) {
    for (size_t y = 0; y < height(); y++) {
      result(y, x) = operator()(x, y);
    }
  }
  return result;
}

double Matrix::norm() const {
  double sum = 0.0;
  for (size_t x = 0; x < width(); x++) {
    for (size_t y = 0; y < height(); y++) {
      double val = operator()(x, y);
      sum += val * val;
    }
  }
  return sqrt(sum);
}

const Matrix Matrix::operator+(const Matrix &other) const {
  Matrix mat(other);
  if (width() != other.width() || height() != other.height()) {
    throw std::runtime_error(
        "Tried to add a " + std::to_string(width()) + "x" +
        std::to_string(height()) + " to a " + std::to_string(mat.width()) +
        "x" + std::to_string(mat.height()) + "; dimensions must match.");
  }
  for (size_t x = 0; x < mat.width(); x++) {
    for (size_t y = 0; y < mat.height(); y++) {
      mat(x, y) += operator()(x, y);
    }
  }
  return mat;
}

Matrix &Matrix::operator=(const Matrix &mat) {
  if (m_rois.size() > 0) {
    if (mat.width() != this->width()) {
      throw std::runtime_error(std::to_string(mat.width()) + "!+ " +
                               std::to_string(this->width()) +
                               "; widths must match.");
    }
    if (mat.height() != this->height()) {
      throw std::runtime_error(std::to_string(mat.height()) + "!+ " +
                               std::to_string(this->height()) +
                               "; widths must match.");
    }
  } else {
    m_width = mat.width();
    m_height = mat.height();
    m_storage.resize(m_width * m_height);
  }
  for (size_t y = 0; y < mat.height(); y++) {
    for (size_t x = 0; x < mat.width(); x++) {
      operator()(x, y) = mat(x, y);
    }
  }
  return *this;
}

const Matrix Matrix::operator*(const double &scalar) const {
  Matrix result(width(), height());
  for (size_t x = 0; x < width(); x++) {
    for (size_t y = 0; y < height(); y++) {
      result(x, y) = scalar * operator()(x, y);
    }
  }
  return result;
}

basic_matrix::Matrix operator+(const double &a, const basic_matrix::Matrix &b) {
  return b + a;
}
basic_matrix::Matrix operator-(const double &a, const basic_matrix::Matrix &b) {
  return a + (-1 * b);
}

basic_matrix::Matrix operator/(const double &a, const basic_matrix::Matrix &b) {
  Matrix result = b;
  for (size_t v = 0; v < b.height(); v++) {
    for (size_t u = 0; u < b.width(); u++) {
      result(u, v) = a / result(u, v);
    }
  }
  return result;
}

const Matrix Matrix::operator-(const Matrix &other) const {
  Matrix return_mat = -1.0 * other + *this;
  return return_mat;
}
const Matrix Matrix::operator-() const { return -1 * (*this); }

Matrix operator*(const double &a, const basic_matrix::Matrix &b) {
  return b * a;
}

const Matrix Matrix::operator+(const double &scalar) const {
  Matrix result = *this;
  for (size_t u = 0; u < this->width(); u++) {
    for (size_t v = 0; v < this->height(); v++) {
      result(u, v) += scalar;
    }
  }
  return result;
}

const Matrix Matrix::operator-(const double &scalar) const {
  return operator+(-scalar);
}
const Matrix Matrix::operator/(const double &scalar) const {
  return (1. / scalar) * (*this);
}

const Matrix Matrix::operator*(const Matrix &other) const {
  if (width() != other.height()) {
    throw std::runtime_error("Tried to mutiply a " + std::to_string(width()) +
                             "x" + std::to_string(height()) + " to a " +
                             std::to_string(other.width()) + "x" +
                             std::to_string(other.height()) +
                             "; condition width1 == height2 must be met.");
  }
  Matrix result(other.width(), height());
  for (size_t this_y = 0; this_y < height(); this_y++) {
    for (size_t other_x = 0; other_x < other.width(); other_x++) {
      double sum = 0.;
      for (size_t this_x = 0; this_x < width(); this_x++) {
        size_t other_y = this_x;
        sum += operator()(this_x, this_y) * other(other_x, other_y);
      }
      size_t dst_x = other_x;
      size_t dst_y = this_y;
      result(dst_x, dst_y) = sum;
    }
  }
  return result;
}

void Matrix::operator+=(const double &scalar) {
  for (size_t v = 0; v < this->height(); v++) {
    for (size_t u = 0; u < this->width(); u++) {
      this->operator()(u, v) += scalar;
    }
  }
}

void Matrix::operator-=(const double &scalar) { (*this) += -scalar; }

void Matrix::operator*=(const double &scalar) {
  for (size_t v = 0; v < this->height(); v++) {
    for (size_t u = 0; u < this->width(); u++) {
      this->operator()(u, v) *= scalar;
    }
  }
}

void Matrix::operator/=(const double &scalar) { (*this) *= (1. / scalar); }

void Matrix::operator+=(const Matrix &matrix) {
  for (size_t v = 0; v < this->height(); v++) {
    for (size_t u = 0; u < this->width(); u++) {
      this->operator()(u, v) += matrix(u, v);
    }
  }
}

void Matrix::operator-=(const Matrix &matrix) {
  for (size_t v = 0; v < this->height(); v++) {
    for (size_t u = 0; u < this->width(); u++) {
      this->operator()(u, v) -= matrix(u, v);
    }
  }
}

const Matrix Matrix::concatRight(const Matrix &other) const {
  if (height() != other.height()) {
    throw std::runtime_error("Matrices should have the same height (to "
                             "concatenate to the right), but don't:" +
                             std::to_string(height()) +
                             "!=" + std::to_string(other.height()));
  }
  Matrix mat(width() + other.width(), height());
  for (size_t y = 0; y < mat.height(); y++) {
    for (size_t x = 0; x < mat.width(); x++) {
      if (x < width()) {
        mat(x, y) = operator()(x, y);
      } else {
        size_t adjusted_x = x - width();
        mat(x, y) = other(adjusted_x, y);
      }
    }
  }
  return mat;
}

const Matrix Matrix::concatDown(const Matrix &other) const {
  if (width() != other.width()) {
    throw std::runtime_error("Matrices should have the same height (to "
                             "concatenate down), but don't:" +
                             std::to_string(width()) +
                             "!=" + std::to_string(other.width()));
  }
  Matrix mat(width(), height() + other.height());
  for (size_t x = 0; x < mat.width(); x++) {
    for (size_t y = 0; y < mat.height(); y++) {
      if (y < height()) {
        mat(x, y) = operator()(x, y);
      } else {
        size_t adjusted_y = y - height();
        mat(x, y) = other(x, adjusted_y);
      }
    }
  }
  return mat;
}

void Matrix::swapRows(const size_t i, const size_t j) {
  if (i >= height()) {
    throw std::out_of_range("Required " + std::to_string(i) + "<" +
                            std::to_string(height()));
  }
  if (j >= height()) {
    throw std::out_of_range("Required " + std::to_string(j) + "<" +
                            std::to_string(height()));
  }
  for (size_t x = 0; x < width(); x++) {
    double temp = operator()(x, i);
    operator()(x, i) = operator()(x, j);
    operator()(x, j) = temp;
  }
}
void Matrix::swapCols(const size_t i, const size_t j) {
  if (i >= width()) {
    throw std::out_of_range("Required " + std::to_string(i) + "<" +
                            std::to_string(width()));
  }
  if (j >= width()) {
    throw std::out_of_range("Required " + std::to_string(j) + "<" +
                            std::to_string(width()));
  }
  for (size_t y = 0; y < width(); y++) {
    double temp = operator()(i, y);
    operator()(i, y) = operator()(j, y);
    operator()(j, y) = temp;
  }
}

const Matrix Matrix::dot(const Matrix &other) const {
  if (width() != other.width() || height() != other.height()) {
    throw std::runtime_error(
        "Dot product does not make sense for matrices of dimensions " +
        std::to_string(width()) + "x" + std::to_string(height()) + " and " +
        std::to_string(other.width()) + "x" + std::to_string(other.height()) +
        "; dimensions must be identical.");
  }
  Matrix mat(width(), height());
  for (size_t x = 0; x < width(); x++) {
    for (size_t y = 0; y < height(); y++) {
      mat(x, y) = operator()(x, y) * other(x, y);
    }
  }
  return mat;
}

double bruteForceDeterminant(const Matrix &mat) {
  if (mat.width() != mat.height()) {
    throw std::runtime_error(
        "determinant only makes sense for square matrices; w=" +
        std::to_string(mat.width()) + ",h=" + std::to_string(mat.height()));
  }
  if (mat.width() == 0) {
    return 0;
  } else if (mat.width() == 1) {
    return mat(0, 0);
  } else if (mat.width() == 2) {
    return mat(0, 0) * mat(1, 1) - mat(1, 0) * mat(0, 1);
  }
  double sum = 0.0;
  Matrix wrapped;
  for (size_t x = 0; x < mat.width(); x++) {
    double sign = (x % 2 == 0) ? 1.0 : -1.0;
    MatrixROI roi_before(0, 1, x, mat.height() - 1, const_cast<Matrix *>(&mat));
    Matrix wrapped(roi_before);
    MatrixROI roi_after(x + 1, 1, mat.width() - x - 1, mat.height() - 1,
                        const_cast<Matrix *>(&mat), x, 0);
    wrapped.addROI(roi_after);
    sum += sign * mat(x, 0) * bruteForceDeterminant(wrapped);
  }
  return sum;
}

/// Implementation of the inner product
} // namespace basic_matrix
