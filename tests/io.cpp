#include "io.hpp"
#include "test_helpers.hpp"

using namespace basic_matrix;

namespace {
inline std::string generateTestDataWithoutNewline() {
  return "0.3,2e3,3.2e3\n999,.4,-3\n1.5e-3,2,1e-3";
}
inline std::string generateTestDataWithNewline() {
  return "0.3,2e3,3.2e3\n999,.4,-3\n1.5e-3,2,1e-3";
}
}; // namespace

void testStringLoading() {
  Matrix expected = {{0.3, 2e3, 3.2e3}, {999, 0.4, -3}, {1.5e-3, 2, 1e-3}};
  {
    Matrix matrix = parseFromString(generateTestDataWithoutNewline());
    ASSERT_EQ(matrix.width(), expected.width());
    ASSERT_EQ(matrix.height(), expected.height());
    double error = (matrix - expected).norm();
    ASSERT_NEAR(error, 0);
  }
  {
    Matrix matrix = parseFromString(generateTestDataWithNewline());
    ASSERT_EQ(matrix.width(), expected.width());
    ASSERT_EQ(matrix.height(), expected.height());
    double error = (matrix - expected).norm();
    ASSERT_NEAR(error, 0);
  }
}

void testFileLoading() {
  Matrix matrix = loadFromFile("tests/matrix.txt");
  ASSERT_EQ(matrix.width(), 3);
  ASSERT_EQ(matrix.height(), 118);
}

int main(int argc, char **argv) {
  testStringLoading();
  testFileLoading();
}
