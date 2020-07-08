#include "standard_functions.hpp"
#include "io.hpp"
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

void testLog() {
  Matrix A = randomMatrix(5, 5, -5, 5);
  {
    Matrix logA = log(A);
    for (size_t v = 0; v < A.height(); v++) {
      for (size_t u = 0; u < A.width(); u++) {
        ASSERT_NEAR(std::log(A(u, v)), logA(u, v));
      }
    }
  }
  {
    Matrix logA = A;
    in_place_log(logA);
    for (size_t v = 0; v < A.height(); v++) {
      for (size_t u = 0; u < A.width(); u++) {
        ASSERT_NEAR(std::log(A(u, v)), logA(u, v));
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
void testLogisticRegressionObjective() {
  Matrix Xy =
      loadFromFile("tests/logistic_regression_linear_decision_boundary.txt");
  Matrix X(3, Xy.height());
  Matrix x_first_two_columns(MatrixROI(1, 0, 2, Xy.height(), &X));
  x_first_two_columns = Matrix(MatrixROI(0, 0, 2, Xy.height(), &Xy));
  for (size_t i = 0; i < Xy.height(); i++) {
    X(0, i) = 1;
  }
  Matrix y = Matrix(MatrixROI(2, 0, 1, Xy.height(), &Xy));
  Matrix theta(1, 3);
  double lambda = 0.;
  Matrix E_out(1, Xy.height());
  LogisticRegressionObjective fctn;
  fctn.energy(theta, X, y, E_out);
  ASSERT_TOL(0.69315, E_out(0, 0), 1e-3);
  Matrix y_predicted(1, Xy.height());
  fctn.eval(theta, X, y_predicted);
  for (size_t v = 0; v < y_predicted.height(); v++) {
    for (size_t u = 0; u < y_predicted.width(); u++) {
      ASSERT_NEAR(y_predicted(u, v), 0.5);
    }
  }
}

int main(int argc, char **argv) {
  testExp();
  testLog();
  testSigmoid();
  testLogisticRegressionObjective();
}
