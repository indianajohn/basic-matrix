#include "standard_functions.hpp"
#include <iostream>

#define IMPL_UTIL_FUNC(name, body)                                             \
  void in_place_##name(Matrix &output) body Matrix name(const Matrix &input) { \
    Matrix output = input;                                                     \
    body return output;                                                        \
  }

namespace basic_matrix {
IMPL_UTIL_FUNC(exp, {
  for (size_t v = 0; v < output.height(); v++) {
    for (size_t u = 0; u < output.width(); u++) {
      output(u, v) = std::exp(output(u, v));
    }
  }
})

IMPL_UTIL_FUNC(log, {
  for (size_t v = 0; v < output.height(); v++) {
    for (size_t u = 0; u < output.width(); u++) {
      output(u, v) = std::log(output(u, v));
    }
  }
})

// 1 / (1 + exp(-input))
IMPL_UTIL_FUNC(sigmoid, {
  output *= -1;
  // = -input
  in_place_exp(output);
  // = exp(-input)
  output += 1;
  // = 1+exp(-input)
  for (size_t v = 0; v < output.height(); v++) {
    for (size_t u = 0; u < output.width(); u++) {
      output(u, v) = 1. / output(u, v);
    }
  }
  // = 1 / (1+exp(-input))
})

double LogisticRegressionObjective::energy(const Matrix &theta, const Matrix &X,
                                           const Matrix &y) {
  // h = sigmoid(X*theta)
  // J = 1/m * (-y'*log(h) - (1 - y)'*log(1-h));
  // J += lambda / (2*m) * theta'*theta
  Matrix h = X * theta;
  double m = static_cast<double>(X.height());
  in_place_sigmoid(h);
  Matrix E_out = (1.0 / m) * (-y.transposeROI() * log(h) -
                              (1.0 - y).transposeROI() * log(1.0 - h));
  // E_out += lambda / (2.0 * m) * theta.transposeROI() * theta;
  return E_out(0, 0);
}

void LogisticRegressionObjective::eval(const Matrix &theta, const Matrix &X,
                                       Matrix &out_y) {
  out_y = X * theta;
  in_place_sigmoid(out_y);
}

void LogisticRegressionObjective::gradient(const Matrix &theta, const Matrix &X,
                                           const Matrix &y, Matrix &J) {
  double m = static_cast<double>(X.height());
  Matrix y_estimated;
  this->eval(theta, X, y_estimated);
  Matrix residual = y_estimated - y;
  J = (1 / m) * X.transposeROI() * residual + (this->lambda / m) * theta;
  J = J.transpose();
}

}; // namespace basic_matrix
