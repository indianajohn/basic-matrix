#include "io.hpp"
#include "matrix_helpers.hpp"
#include "test_helpers.hpp"
#include <cstdio>

using namespace basic_matrix;

namespace {
inline std::string generateTestDataWithoutNewline() {
  return "0.3,2e3,3.2e3\n999,.4,-3\n1.5e-3,2,1e-3";
}
inline std::string generateTestDataWithNewline() {
  return "0.3,2e3,3.2e3\n999,.4,-3\n1.5e-3,2,1e-3\n";
}
}; // namespace

void testStringLoading() {
  {
    Matrix expected = {{0.3, 2e3, 3.2e3}, {999, 0.4, -3}, {1.5e-3, 2, 1e-3}};
    Matrix matrix = parseFromString(generateTestDataWithoutNewline());
    assertMatrixNear(matrix, expected);
  }
  {
    Matrix expected = {{0.3, 2e3, 3.2e3}, {999, 0.4, -3}, {1.5e-3, 2, 1e-3}};
    Matrix matrix = parseFromString(generateTestDataWithNewline());
    assertMatrixNear(matrix, expected);
  }
  {
    Matrix matrix = parseFromString("");
    Matrix expected = {};
    assertMatrixNear(matrix, expected);
  }
  {
    Matrix matrix = parseFromString("3");
    Matrix expected = {3};
    assertMatrixNear(matrix, expected);
  }
  {
    Matrix matrix = parseFromString("3,4.0");
    Matrix expected = {3, 4.0};
    assertMatrixNear(matrix, expected);
  }
  {
    Matrix matrix = parseFromString("3\n4.0");
    std::vector<std::vector<double>> expected = {{3.}, {4.0}};
    assertMatrixNear(matrix, Matrix(expected));
  }
}

void testFileLoading() {
  Matrix matrix = loadFromFile("tests/matrix.txt");
  ASSERT_EQ(matrix.width(), 3);
  ASSERT_EQ(matrix.height(), 118);
}

void testStringWriting() {
  for (size_t trial = 0; trial < 20; trial++) {
    Matrix random_mat = randomMatrix(rand() % 10, rand() % 10, -20.0, 20.0);
    std::string mat_string = writeToString(random_mat);
    Matrix round_trip = parseFromString(mat_string);
    if (random_mat.width() > 0 && random_mat.height() > 0) {
      assertMatrixNear(random_mat, round_trip, 1e-4);
    }
  }
}

void testFileWriting() {
  // Assume path is set
  std::string path = "temp_mat.txt";
  for (size_t trial = 0; trial < 20; trial++) {
    Matrix random_mat = randomMatrix(rand() % 10, rand() % 10, -20.0, 20.0);
    writeToFile(path, random_mat);
    Matrix round_trip = loadFromFile(path);
    if (random_mat.width() > 0 && random_mat.height() > 0) {
      assertMatrixNear(random_mat, round_trip, 1e-4);
    }
    std::remove(path.c_str());
  }
}

int main(int argc, char **argv) {
  testStringLoading();
  testFileLoading();
  testStringWriting();
  testFileWriting();
}
