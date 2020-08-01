#include "matrix.hpp"
#include "matrix_helpers.hpp"
#include "test_helpers.hpp"
#include <chrono>
#include <math.h>
#include <random>

using namespace basic_matrix;

void transposeWorks() {
  Matrix mat({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}});
  Matrix transposed = mat.transpose();
  ASSERT_EQ(mat.width(), transposed.height());
  ASSERT_EQ(mat.height(), transposed.width());
  for (size_t x = 0; x < mat.width(); x++) {
    for (size_t y = 0; y < mat.height(); y++) {
      ASSERT_EQ(mat(x, y), transposed(y, x));
    }
  }
}

void normWorks() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10.0, 10.0);
  double sum = 0.0;
  size_t w = 15;
  size_t h = 20;
  Matrix mat(w, h);
  for (size_t x = 0; x < w; x++) {
    for (size_t y = 0; y < h; y++) {
      double val = dis(rd);
      mat(x, y) = val;
      sum += val * val;
    }
  }
  double norm = sqrt(sum);
  ASSERT_NEAR(norm, mat.norm());
}

void addWorks() {
  {
    Matrix mat1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    Matrix mat2({{7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}});
    Matrix expected({{8.0, 10.0, 12.0}, {14.0, 16.0, 18.0}});
    Matrix mat3 = mat1 + mat2;
    assertMatrixNear(expected, mat3);
  }
  {
    Matrix mat1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    Matrix mat2({{7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}});
    Matrix expected({{8.0, 10.0, 12.0}, {14.0, 16.0, 18.0}});
    mat1 += mat2;
    assertMatrixNear(expected, mat1);
  }
}

void subtractWorks() {
  {
    Matrix mat1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    Matrix mat2({{7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}});
    Matrix expected({{6.0, 6.0, 6.0}, {6.0, 6.0, 6.0}});
    Matrix mat3 = mat2 - mat1;
    assertMatrixNear(expected, mat3);
  }
  {
    Matrix mat1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    Matrix mat2({{7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}});
    Matrix expected({{6.0, 6.0, 6.0}, {6.0, 6.0, 6.0}});
    mat2 -= mat1;
    assertMatrixNear(expected, mat2);
  }
}

void multiplyWorks() {
  Matrix mat1(
      {{1.1, 2.2, 3.3, 4.4}, {5.5, 6.6, 7.7, 8.8}, {9.9, 10.1, 11.11, 12.12}});
  Matrix mat2({{13.13, 14.14, 15.15, 16.16},
               {17.17, 18.18, 19.19, 20.2},
               {21.21, 22.22, 23.23, 24.24}});
  Matrix mat_expected({{166.65, 211.09, 255.53},
                       {424.40, 539.95, 655.49},
                       {636.98, 811.63, 986.28}});
  Matrix mat_result = mat1 * mat2.transpose();
  assertMatrixNear(mat_expected, mat_result, 1e-2);
}

void concatRightWorks() {
  Matrix mat1({{1.1, 2.2}, {5.5, 6.6}, {9.9, 10.1}});
  Matrix mat2({{13.13, 14.14, 15.15, 16.16},
               {17.17, 18.18, 19.19, 20.2},
               {21.21, 22.22, 23.23, 24.24}});
  Matrix mat_expected({{1.1, 2.2, 13.13, 14.14, 15.15, 16.16},
                       {5.5, 6.6, 17.17, 18.18, 19.19, 20.2},
                       {9.9, 10.1, 21.21, 22.22, 23.23, 24.24}});
  Matrix mat_result = mat1.concatRight(mat2);
  assertMatrixNear(mat_expected, mat_result);
}

void concatDownWorks() {
  Matrix mat1({{1.1, 2.2, 3.3, 4.4}, {5.5, 6.6, 7.7, 8.8}});
  Matrix mat2({{13.13, 14.14, 15.15, 16.16},
               {17.17, 18.18, 19.19, 20.2},
               {21.21, 22.22, 23.23, 24.24}});
  Matrix mat_expected({{13.13, 14.14, 15.15, 16.16},
                       {17.17, 18.18, 19.19, 20.2},
                       {21.21, 22.22, 23.23, 24.24},
                       {1.1, 2.2, 3.3, 4.4},
                       {5.5, 6.6, 7.7, 8.8}});
  Matrix mat_result = mat2.concatDown(mat1);
  assertMatrixNear(mat_expected, mat_result);
}

void dotProductWorks() {
  {
    Matrix v1({5, 4, 3});
    Matrix v2({5, 4, 3});
    Matrix v3 = v1.dot(v2);
    Matrix v3_expected({25, 16, 9});
    assertMatrixNear(v3, v3_expected);
  }
}

void swapRowsWorks() {
  Matrix mat_result({{13.13, 14.14, 15.15, 16.16},
                     {17.17, 18.18, 19.19, 20.2},
                     {21.21, 22.22, 23.23, 24.24}});
  mat_result.swapRows(0, 1);
  Matrix mat_expected({{17.17, 18.18, 19.19, 20.2},
                       {13.13, 14.14, 15.15, 16.16},
                       {21.21, 22.22, 23.23, 24.24}});
  assertMatrixNear(mat_expected, mat_result);
}

void swapColsWorks() {
  Matrix mat_result({{13.13, 14.14, 15.15, 16.16},
                     {17.17, 18.18, 19.19, 20.2},
                     {21.21, 22.22, 23.23, 24.24}});
  Matrix mat_expected({{13.13, 14.14, 16.16, 15.15},
                       {17.17, 18.18, 20.2, 19.19},
                       {21.21, 22.22, 24.24, 23.23}});
  mat_result.swapCols(2, 3);
}

void scalarMultiplyWorks() {
  Matrix A({{0.5, -1.4, 5.5}, {4.53, -2.593, 2.21}});
  Matrix minus_A_expectedA({{-0.5, 1.4, -5.5}, {-4.53, 2.593, -2.21}});
  Matrix minus_A = -1.0 * A;
  Matrix minus_A_post = A * -1.0;
  assertMatrixNear(minus_A_expectedA, minus_A);
  assertMatrixNear(minus_A_expectedA, minus_A_post);
  A *= -1.0;
  assertMatrixNear(minus_A_expectedA, A);
}

void minusWorks() {
  Matrix A({{0.5, -1.4, 5.5}, {4.53, -2.593, 2.21}});
  Matrix A_2({{0.5, -1.4, 5.5}, {4.53, -2.593, 2.21}});
  Matrix zero(3, 2);
  Matrix A_minus_A_2 = A - A_2;
  assertMatrixNear(zero, A_minus_A_2);
}

void scalarPlusWorks() {
  {
    Matrix A({{0.5, -1.4, 5.5}, {4.53, -2.593, 2.21}});
    Matrix A_expected({{1.5, -0.4, 6.5}, {5.53, -1.593, 3.21}});
    Matrix A_plus_one = A + 1;
    assertMatrixNear(A_plus_one, A_expected);
  }
  {
    Matrix A({{0.5, -1.4, 5.5}, {4.53, -2.593, 2.21}});
    Matrix A_expected({{1.5, -0.4, 6.5}, {5.53, -1.593, 3.21}});
    Matrix A_plus_one = 1 + A;
    // assertMatrixNear(A_plus_one, A_expected);
  }
  {
    Matrix A({{0.5, -1.4, 5.5}, {4.53, -2.593, 2.21}});
    Matrix A_expected({{1.5, -0.4, 6.5}, {5.53, -1.593, 3.21}});
    A += 1.;
    // assertMatrixNear(A, A_expected);
  }
}

void scalarMinusWorks() {
  {
    Matrix A({{0.5, -1.4, 5.5}, {4.53, -2.593, 2.21}});
    Matrix A_expected({{-0.5, -2.4, 4.5}, {3.53, -3.593, 1.21}});
    Matrix A_minus_one = A - 1;
    assertMatrixNear(A_minus_one, A_expected);
  }
  {
    Matrix A({{0.5, -1.4, 5.5}, {4.53, -2.593, 2.21}});
    Matrix A_expected({{0.5, 2.4, -4.5}, {-3.53, 3.593, -1.21}});
    Matrix one_minus_A = 1 - A;
    assertMatrixNear(one_minus_A, A_expected);
  }
  {
    Matrix A({{0.5, -1.4, 5.5}, {4.53, -2.593, 2.21}});
    Matrix A_expected({{-0.5, -2.4, 4.5}, {3.53, -3.593, 1.21}});
    A -= 1.;
    assertMatrixNear(A, A_expected);
  }
}

void scalarDivideWorks() {
  {
    Matrix A({{0.5, -1.4, 5.5}, {4.53, -2.593, 2.21}});
    Matrix A_expected({{0.25, -0.7, 2.75}, {2.265, -1.2965, 1.105}});
    Matrix A_div_2 = A / 2.;
    assertMatrixNear(A_div_2, A_expected);
  }
  {
    Matrix A({{0.5, -1.4, 5.5}, {4.53, -2.593, 2.21}});
    Matrix A_expected({{4.0, -1.42857142857, 0.363636363636},
                       {0.441501103753, -0.771307365985, 0.904977375566}});
    Matrix A_div_2 = 2 / A;
    assertMatrixNear(A_div_2, A_expected);
  }
  {
    Matrix A({{0.5, -1.4, 5.5}, {4.53, -2.593, 2.21}});
    Matrix A_expected({{0.25, -0.7, 2.75}, {2.265, -1.2965, 1.105}});
    A /= 2.;
    assertMatrixNear(A, A_expected);
  }
}

void negationWorks() {
  Matrix A({{0.5, -1.4, 5.5}, {4.53, -2.593, 2.21}});
  Matrix A_expected({{-0.5, 1.4, -5.5}, {-4.53, 2.593, -2.21}});
  assertMatrixNear(-A, A_expected);
}

void simdWorks() {
  for (int trial = 0; trial < 50; trial++) {
    int h = rand() % 100 + 2;
    Matrix A = randomMatrix(h, h, -100.0, 100.0);
    Matrix B = randomMatrix(h, h, -100.0, 100.0);
    Matrix C_naive(h, h);
    Matrix C_simd(h, h);
    naiveMultiply(A, B, C_naive);
    simdMultiply(A, B, C_simd);
    assertMatrixNear(C_naive, C_simd);
  }
}

void benchmark() {
  int h = 130;
  Matrix A = randomMatrix(h, h, -100.0, 100.0);
  Matrix B = randomMatrix(h, h, -100.0, 100.0);
  auto start = std::chrono::system_clock::now();
  for (size_t trial = 0; trial < 50; trial++) {
    Matrix C(h, h);
    simdMultiply(A, B, C);
  }
  int simd_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now() - start)
                    .count();
  start = std::chrono::system_clock::now();
  for (size_t trial = 0; trial < 10; trial++) {
    Matrix C(h, h);
    naiveMultiply(A, B, C);
  }
  int naive_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now() - start)
                     .count();
  // In testing, simd was 10 times faster.
  ASSERT(2 * simd_ms < naive_ms);
}

int main() {
  transposeWorks();
  normWorks();
  addWorks();
  minusWorks();
  subtractWorks();
  multiplyWorks();
  concatRightWorks();
  concatDownWorks();
  swapRowsWorks();
  swapColsWorks();
  dotProductWorks();
  scalarMultiplyWorks();
  scalarPlusWorks();
  scalarMinusWorks();
  scalarDivideWorks();
  negationWorks();
  simdWorks();
  benchmark();
}
