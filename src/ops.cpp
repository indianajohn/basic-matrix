#include "matrix.hpp"
#include <iostream>
#include <math.h>
#include <random>
#include <stdexcept>
#include <stdint.h>
#include <string.h>

namespace basic_matrix {
namespace {

/// 64-bit float vector
typedef double float8 __attribute__((vector_size(64)));

/// Loads a simd float8 value from a pointer.
template <typename T> float8 loadFloat8(const T *ptr) {
  return *((float8 *)ptr);
}

/// Stores the simd float8 value to a pointer.
template <typename T> void storeFloat8(T *ptr, const float8 &val) {
  *((float8 *)(ptr)) = val;
}

/// Accumulate (+=) the simd float8 value to a pointer.
template <typename T> void addFloat8(T *ptr, const float8 &val) {
  *((float8 *)(ptr)) += val;
}

/// Broadcast the input value to a float8.
template <typename T> float8 broadcastFloat8(const T &val) {
  return val - (float8){};
}

/// Perform an unaligned load of a float8 from a float pointer.
inline float8 loadUnalignedFloat8(const double *p) {
  float8 res;
  memcpy(&res, p, sizeof(float8));
  return res;
}

/// Perform an unaligned store to a float pointer.
inline void storeUnalignedFloat8(double *p, float8 v) {
  memcpy(p, &v, sizeof(float8));
}

/// Perform an unaligned addition to a float pointer.
inline void AdduFloat8(double *p, float8 v) {
  storeUnalignedFloat8(p, loadUnalignedFloat8(p) + v);
}

/// Naive matrix multiplication on a block.
inline void naiveMatmul(const Matrix &M1, const Matrix &M2, Matrix &out, int m,
                        int n, int block_width, const int &u_offset,
                        const int &v_offset, const int &v_dst) {
  double *c = &out(v_dst, v_offset);
  int ldc = out.width();
  const double *a = &M1(u_offset, v_offset);
  int lda = M1.width();
  const double *b = &M2(v_dst, u_offset);
  int ldb = M2.width();
  for (int p = 0; p < block_width; p++) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        c[i * ldc + j] += a[i * lda + p] * b[p * ldb + j];
      }
    }
  }
}

/// Compute a 4x16 block of C using a vectorized dot product.
inline void dot4x16(const Matrix &M1, const Matrix &M2, Matrix &out,
                    const int &u_offset, const int &v_offset, const int &j,
                    const int &block_width) {
  float8 ctmp07[4] = {0.0};
  float8 ctmp815[4] = {0.0};
  for (int p = 0; p < block_width; p++) {
    float8 a0p = broadcastFloat8(M1(p + u_offset, 0 + v_offset));
    float8 a1p = broadcastFloat8(M1(p + u_offset, 1 + v_offset));
    float8 a2p = broadcastFloat8(M1(p + u_offset, 2 + v_offset));
    float8 a3p = broadcastFloat8(M1(p + u_offset, 3 + v_offset));
    float8 bp0p7 = loadUnalignedFloat8(&M2(0 + j, p + u_offset));
    float8 bp8p15 = loadUnalignedFloat8(&M2(8 + j, p + u_offset));
    ctmp07[0] += a0p * bp0p7;
    ctmp07[1] += a1p * bp0p7;
    ctmp07[2] += a2p * bp0p7;
    ctmp07[3] += a3p * bp0p7;
    ctmp815[0] += a0p * bp8p15;
    ctmp815[1] += a1p * bp8p15;
    ctmp815[2] += a2p * bp8p15;
    ctmp815[3] += a3p * bp8p15;
  }
  AdduFloat8(&out(0 + j, 0 + v_offset), ctmp07[0]);
  AdduFloat8(&out(0 + j, 1 + v_offset), ctmp07[1]);
  AdduFloat8(&out(0 + j, 2 + v_offset), ctmp07[2]);
  AdduFloat8(&out(0 + j, 3 + v_offset), ctmp07[3]);
  AdduFloat8(&out(8 + j, 0 + v_offset), ctmp815[0]);
  AdduFloat8(&out(8 + j, 1 + v_offset), ctmp815[1]);
  AdduFloat8(&out(8 + j, 2 + v_offset), ctmp815[2]);
  AdduFloat8(&out(8 + j, 3 + v_offset), ctmp815[3]);
}

/// Tiled matrix multiplication. Multiply 4x16 blocks until that becomes
/// impossible, and finish off by mutiplying the edges conventionally.
inline void tiledMatrixMult(const Matrix &M1, const Matrix &M2, Matrix &out,
                            const int &u_offset, const int &block_width,
                            const int &v_offset, const int &block_height) {
  constexpr int tile_height = 4;
  constexpr int tile_width = 16;
  int out_width = out.width();
  // Multiply 4x16 blocks until we run out of them.
  for (int j = 0; j < out_width - tile_width + 1; j += tile_width) {
    for (int i = 0; i < block_height - tile_height + 1; i += tile_height) {
      dot4x16(M1, M2, out, 0 + u_offset, i + v_offset, j, block_width);
    }
  }

  // multiply 3 three submatrices to the right, bottom, and bottom-right of
  // the 4x16 regions.
  int i = (block_height / tile_height) * tile_height;
  int j = (out_width / tile_width) * tile_width;
  if (i < block_height) {
    // Lower-left block
    naiveMatmul(M1, M2, out, block_height - i, j, block_width, 0 + u_offset,
                i + v_offset, 0);
  }
  if (j < out_width) {
    // Upper-right block
    naiveMatmul(M1, M2, out, i, out_width - j, block_width, 0 + u_offset,
                0 + v_offset, j);
  }
  if (i < block_height && j < out_width) {
    // Lower-right block
    naiveMatmul(M1, M2, out, block_height - i, out_width - j, block_width,
                0 + u_offset, i + v_offset, j);
  }
}
}; // namespace

void naiveMultiply(const Matrix &A, const Matrix &B, Matrix &C) {
  for (size_t this_y = 0; this_y < A.height(); this_y++) {
    for (size_t other_x = 0; other_x < B.width(); other_x++) {
      double sum = 0.;
      for (size_t this_x = 0; this_x < A.width(); this_x++) {
        size_t other_y = this_x;
        sum += A(this_x, this_y) * B(other_x, other_y);
      }
      size_t dst_x = other_x;
      size_t dst_y = this_y;
      C(dst_x, dst_y) = sum;
    }
  }
}

void simdMultiply(const Matrix &M1, const Matrix &M2, Matrix &out) {
  // tile size
  constexpr size_t mc = 256;
  constexpr size_t kc = 128;
  for (size_t p = 0; p < M1.width(); p += kc) {
    // full size if there's room left, otherwise the remaining width.
    int block_width = std::min(M1.width() - p, kc);
    for (size_t i = 0; i < out.height(); i += mc) {
      int block_height = std::min(out.height() - i, mc);
      tiledMatrixMult(M1, M2, out, p, block_width, i, block_height);
    }
  }
}

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
    if (m_width != mat.width() || m_height != mat.height()) {
      m_width = mat.width();
      m_height = mat.height();
      m_storage.resize(m_width * m_height);
    }
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
  if (this->contiguous() && other.contiguous()) {
    simdMultiply(*this, other, result);
  } else {
    naiveMultiply(*this, other, result);
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

} // namespace basic_matrix
