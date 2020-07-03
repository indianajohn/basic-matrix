#include "standard_functions.hpp"
#include "matrix_helpers.hpp"
using namespace basic_matrix;

void testExp() {
  Matrix A = randomMatrix(5, 5, -5, 5);
  {
    Matrix expA = exp(A);
    for (size_t v = 0; v < A.height(); v++) {
      for (size_t u = 0; u < A.width(); u++) {
        ASSERT_NEAR(std::exp(A(u, v)), expA(u, v));
      }
    }
  }
  {
    Matrix expA = A;
    in_place_exp(expA);
    for (size_t v = 0; v < A.height(); v++) {
      for (size_t u = 0; u < A.width(); u++) {
        ASSERT_NEAR(std::exp(A(u, v)), expA(u, v));
      }
    }
  }
}

void testSigmoid() {
  Matrix A = randomMatrix(5, 5, -5, 5);
  Matrix A_sigmoid = A;
  for (size_t v = 0; v < A_sigmoid.height(); v++) {
    for (size_t u = 0; u < A_sigmoid.width(); u++) {
      double z = A_sigmoid(u, v);
      A_sigmoid(u, v) = 1. / (1. + exp(-z));
    }
  }
  {
    Matrix sigmoid_result = sigmoid(A);
    for (size_t v = 0; v < A.height(); v++) {
      for (size_t u = 0; u < A.width(); u++) {
        ASSERT_NEAR(A_sigmoid(u, v), sigmoid_result(u, v));
      }
    }
  }
  {
    Matrix sigmoid_result = A;
    in_place_sigmoid(sigmoid_result);
    for (size_t v = 0; v < A.height(); v++) {
      for (size_t u = 0; u < A.width(); u++) {
        ASSERT_NEAR(A_sigmoid(u, v), sigmoid_result(u, v));
      }
    }
  }
}

int main(int argc, char **argv) {
  testExp();
  testSigmoid();
}
