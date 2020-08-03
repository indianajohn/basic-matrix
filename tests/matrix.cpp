#include "matrix.hpp"
#include "matrix_helpers.hpp"
#include "test_helpers.hpp"
#include <iostream>
#include <random>
#include <sstream>
using namespace basic_matrix;

void emptyMatrix() {
  Matrix mat;
  ASSERT_EQ(mat.width(), 0);
  ASSERT_EQ(mat.height(), 0);
  // make sure ostream operator handles empty matrices well
  std::stringstream ss;
  ss << mat;
}

void initWorks() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(std::numeric_limits<double>::min(),
                                       std::numeric_limits<double>::max());
  for (size_t width = 0; width < 10; width++) {
    for (size_t height = 0; height < 20; height++) {
      Matrix test(width, height);
      ASSERT_EQ(test.width(), width);
      ASSERT_EQ(test.height(), height);
      for (size_t x = 0; x < test.width(); x++) {
        for (size_t y = 0; y < test.height(); y++) {
          ASSERT_NEAR(test(x, y), 0);
          double val = dis(gen);
          test(x, y) = val;
          ASSERT_NEAR(test(x, y), val);
        }
      }
    }
  }
}

void initWithValue() {
  std::vector<std::vector<double>> vec = {{1, 2, 4}, {3.5, 4.2, 1.2}};
  Matrix test(vec);
  ASSERT_EQ(test.width(), 3);
  ASSERT_EQ(test.height(), 2);
  Matrix test2({{2.320359359, 9.2, 0.3}});
  ASSERT_EQ(test2.width(), 3);
  ASSERT_EQ(test2.height(), 1);
  Matrix test3 = {{0.5, 10.4, 4.3}, {0.2, 4.4, 1.1}};
  ASSERT_EQ(test3.width(), 3);
  ASSERT_EQ(test3.height(), 2);
  Matrix test4 = {0.3, 0.5, 0.19, 0.55};
  ASSERT_EQ(test4.height(), 1);
  ASSERT_EQ(test4.width(), 4);
}

void boundingBoxWorks() {
  BoundingBox bb;
  bb.extend(std::make_pair(1, 1));
  ASSERT_EQ(bb.minX(), 1);
  ASSERT_EQ(bb.minY(), 1);
  ASSERT_EQ(bb.maxX(), 1);
  ASSERT_EQ(bb.maxY(), 1);
  bb.extend(std::make_pair(5, 2));
  ASSERT_EQ(bb.minX(), 1);
  ASSERT_EQ(bb.minY(), 1);
  ASSERT_EQ(bb.maxX(), 5);
  ASSERT_EQ(bb.maxY(), 2);
  bb.extend(std::make_pair(5, 1));
  ASSERT_EQ(bb.minX(), 1);
  ASSERT_EQ(bb.minY(), 1);
  ASSERT_EQ(bb.maxX(), 5);
  ASSERT_EQ(bb.maxY(), 2);
  bb.extend(std::make_pair(5, 4));
  ASSERT_EQ(bb.minX(), 1);
  ASSERT_EQ(bb.minY(), 1);
  ASSERT_EQ(bb.maxX(), 5);
  ASSERT_EQ(bb.maxY(), 4);
  bb.extend(std::make_pair(6, 3));
  ASSERT_EQ(bb.minX(), 1);
  ASSERT_EQ(bb.minY(), 1);
  ASSERT_EQ(bb.maxX(), 6);
  ASSERT_EQ(bb.maxY(), 4);
  bb.extend(std::make_pair(7, 8));
  ASSERT_EQ(bb.minX(), 1);
  ASSERT_EQ(bb.minY(), 1);
  ASSERT_EQ(bb.maxX(), 7);
  ASSERT_EQ(bb.maxY(), 8);
  bb.extend(std::make_pair(0, 0));
  ASSERT_EQ(bb.minX(), 0);
  ASSERT_EQ(bb.minY(), 0);
  ASSERT_EQ(bb.maxX(), 7);
  ASSERT_EQ(bb.maxY(), 8);
}

void roisWork() {
  size_t x_dst, y_dst;
  size_t x_src, y_src;
  // Basic usage
  MatrixROI roi(0, 0, 3, 4, nullptr, 0, 0, false);
  roi.srcToDst(0, 0, x_dst, y_dst);
  ASSERT_EQ(x_dst, 0);
  ASSERT_EQ(y_dst, 0);
  roi.srcToDst(2, 3, x_dst, y_dst);
  ASSERT_EQ(x_dst, 2);
  ASSERT_EQ(y_dst, 3);
  ASSERT_EQ(roi.isInside(2, 3), true);
  ASSERT_EQ(roi.isInside(3, 4), false);

  // offset src
  roi = MatrixROI(1, 1, 3, 4, nullptr, 0, 0, false);
  roi.srcToDst(1, 1, x_dst, y_dst);
  ASSERT_EQ(x_dst, 0);
  ASSERT_EQ(y_dst, 0);
  roi.srcToDst(2, 1, x_dst, y_dst);
  ASSERT_EQ(x_dst, 1);
  ASSERT_EQ(y_dst, 0);
  roi.srcToDst(1, 2, x_dst, y_dst);
  ASSERT_EQ(x_dst, 0);
  ASSERT_EQ(y_dst, 1);
  ASSERT_EQ(roi.isInside(2, 3), true);
  ASSERT_EQ(roi.isInside(3, 4), false);

  // offset dst
  roi = MatrixROI(1, 1, 3, 4, nullptr, 2, 3, false);
  ASSERT_EQ(roi.isInside(0, 0), false);
  ASSERT_EQ(roi.isInside(1, 2), false);
  ASSERT_EQ(roi.isInside(2, 3), true);
  ASSERT_EQ(roi.isInside(3, 3), true);
  ASSERT_EQ(roi.isInside(2, 4), true);
  ASSERT_EQ(roi.isInside(4, 6), true);
  ASSERT_EQ(roi.isInside(5, 6), false);
  ASSERT_EQ(roi.isInside(4, 7), false);
  ASSERT_EQ(roi.isInside(5, 7), false);

  // Transpose
  roi = MatrixROI(1, 1, 3, 4, nullptr, 2, 3, true);
  roi.dstToSrc(2, 3, x_src, y_src);
  ASSERT_EQ(x_src, 1);
  ASSERT_EQ(y_src, 1);
  roi.dstToSrc(3, 3, x_src, y_src);
  ASSERT_EQ(x_src, 1);
  ASSERT_EQ(y_src, 2);
  roi.dstToSrc(2, 4, x_src, y_src);
  ASSERT_EQ(x_src, 2);
  ASSERT_EQ(y_src, 1);
  roi.srcToDst(1, 1, x_dst, y_dst);
  ASSERT_EQ(x_dst, 2);
  ASSERT_EQ(y_dst, 3);
  roi.srcToDst(2, 1, x_dst, y_dst);
  ASSERT_EQ(x_dst, 2);
  ASSERT_EQ(y_dst, 4);
  roi.srcToDst(1, 2, x_dst, y_dst);
  ASSERT_EQ(x_dst, 3);
  ASSERT_EQ(y_dst, 3);
  ASSERT_EQ(roi.isInside(0, 0), false);
  ASSERT_EQ(roi.isInside(2, 1), false);
  ASSERT_EQ(roi.isInside(2, 3), true);
  ASSERT_EQ(roi.isInside(5, 5), true);
  ASSERT_EQ(roi.isInside(6, 5), false);
  ASSERT_EQ(roi.isInside(5, 6), false);
  ASSERT_EQ(roi.isInside(6, 6), false);
}

void wrappedMatricesWork() {
  // Simple test
  {
    Matrix test = {{1, 2, 4}, {3.5, 4.2, 1.2}};
    Matrix wrapped(MatrixROI(0, 0, test.width(), test.height(), &test));
    ASSERT_MATRIX_NEAR(wrapped, test);
    std::stringstream ss;
    // Make sure stream works on wrapped matrices
    ss << wrapped << std::endl;
    wrapped(1, 1) = 5.5;
    ASSERT_NEAR(wrapped(1, 1), 5.5);
    ASSERT_NEAR(wrapped(1, 1), test(1, 1));
  }
  // Wrapping 2 matrices in one empty wrapper
  {
    Matrix test0 = {{1, 2, 4}, {3.5, 4.2, 1.2}};
    Matrix test1 = {{3, 5, 9}};
    Matrix test2;
    Matrix wrapped(
        MatrixROI(0, 0, test0.width(), test0.height(), &test0, 0, 0));
    wrapped.addROI(
        MatrixROI(0, 0, test1.width(), test1.height(), &test1, 0, 2));
    Matrix wrapped_expected = test0.concatDown(test1);
    std::stringstream ss;
    // Make sure stream works on wrapped matrices
    ss << wrapped << std::endl;
    ASSERT_MATRIX_NEAR(wrapped, wrapped_expected);
    wrapped(1, 1) = 5.5;
    ASSERT_NEAR(wrapped(1, 1), 5.5);
    ASSERT_NEAR(wrapped(1, 1), test0(1, 1));
    wrapped(1, 2) = 10.5;
    ASSERT_NEAR(wrapped(1, 2), 10.5);
    ASSERT_NEAR(wrapped(1, 2), test1(1, 0));
  }
  {
    // Transpose
    Matrix test = {{1, 2, 4}, {3.5, 4.2, 1.2}};
    Matrix test_transposed = test.transpose();
    ASSERT_MATRIX_NEAR(test_transposed, test.transposeROI());
  }
  {
    // const transpose ROI
    const Matrix test = {{1, 2, 4}, {3.5, 4.2, 1.2}};
    Matrix test_transposed = test.transpose();
    ASSERT_MATRIX_NEAR(test_transposed, test.transposeROI());
  }
}

void detWorksOn1x1() {
  Matrix mat = {2.2};
  double det = mat.det();
  ASSERT_NEAR(det, 2.2);
}

void detWorksOn2x2() {
  Matrix mat = {{1, 2}, {3, 4}};
  double det = mat.det();
  ASSERT_NEAR(det, -2.0);
}

void detWorksOn3x3() {
  Matrix mat = {{1, 4, 11}, {4, 59, 63}, {7, 18, 9}};
  double det = mat.det();
  ASSERT_NEAR(det, -2734.0);
}

void detWorksOn4x4() {
  Matrix mat = {
      {1, 4, 11, 12}, {4, 59, 63, 64}, {7, 18, 9, 10}, {12, 49, 19, 19}};
  double det = mat.det();
  ASSERT_NEAR(det, -54.0);
}

void equalsWorks() {
  Matrix mat = {
      {1, 4, 11, 12}, {4, 59, 63, 64}, {7, 18, 9, 10}, {12, 49, 19, 19}};
  // Wrapping part of a matrix and using the equals operator to set the
  // submatrix to the result of a calculation is useful in many situations.
  Matrix wrapped(MatrixROI(0, 0, 2, 2, &mat));
  Matrix mat_to_set = {{9, 9}, {10, 10}};
  wrapped = mat_to_set;
  ASSERT_MATRIX_NEAR(wrapped, mat_to_set);
  Matrix expected = {
      {9, 9, 11, 12}, {10, 10, 63, 64}, {7, 18, 9, 10}, {12, 49, 19, 19}};
  ASSERT_MATRIX_NEAR(mat, expected);
}

void rowWorks() {
  Matrix mat = {
      {1, 4, 11, 12}, {4, 59, 63, 64}, {7, 18, 9, 10}, {12, 49, 19, 19}};
  ASSERT_MATRIX_NEAR(mat.row(0), Matrix({1, 4, 11, 12}));
  ASSERT_MATRIX_NEAR(mat.row(1), Matrix({4, 59, 63, 64}));
  ASSERT_MATRIX_NEAR(mat.row(2), Matrix({7, 18, 9, 10}));
  ASSERT_MATRIX_NEAR(mat.row(3), Matrix({12, 49, 19, 19}));
}

void colWorks() {
  Matrix mat = {
      {1, 4, 11, 12}, {4, 59, 63, 64}, {7, 18, 9, 10}, {12, 49, 19, 19}};
  ASSERT_MATRIX_NEAR(mat.col(0), Matrix({1, 4, 7, 12}).transposeROI());
  ASSERT_MATRIX_NEAR(mat.col(1), Matrix({4, 59, 18, 49}).transposeROI());
  ASSERT_MATRIX_NEAR(mat.col(2), Matrix({11, 63, 9, 19}).transposeROI());
  ASSERT_MATRIX_NEAR(mat.col(3), Matrix({12, 64, 10, 19}).transposeROI());
}

void reshapeWorks() {
  Matrix mat = {
      {1, 4, 11, 12}, {4, 59, 63, 64}, {7, 18, 9, 10}, {12, 49, 19, 19}};
  Matrix mat_expected = {{1, 4, 11, 12, 4, 59, 63, 64},
                         {7, 18, 9, 10, 12, 49, 19, 19}};
  mat.reshape(8, 2);
  ASSERT_MATRIX_NEAR(mat, mat_expected);
}

int main() {
  emptyMatrix();
  initWorks();
  initWithValue();
  boundingBoxWorks();
  roisWork();
  wrappedMatricesWork();
  detWorksOn1x1();
  detWorksOn2x2();
  detWorksOn3x3();
  detWorksOn4x4();
  equalsWorks();
  rowWorks();
  colWorks();
  reshapeWorks();
}
