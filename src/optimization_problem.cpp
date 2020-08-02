#include "optimization_problem.hpp"
#include <iostream>

namespace basic_matrix {

void estimateJacobian(const OptimizationProblem &problem, Matrix &J,
                      const double epsilon) {
  // #/outputs of the function
  size_t width = problem.inputs.num_params;
  size_t height = problem.inputs.y.height();
  if (J.width() != width || J.height() != height) {
    J = Matrix(width, height);
  }
  Matrix y0(1, problem.inputs.y.height());
  Matrix y1(1, problem.inputs.y.height());
  for (size_t x_i = 0; x_i < problem.outputs.theta.height(); x_i++) {
    for (size_t y_i = 0; y_i < problem.inputs.X.height(); y_i++) {
      Matrix theta_perturbed = problem.outputs.theta;
      theta_perturbed(0, x_i) -= epsilon;
      problem.inputs.function(theta_perturbed,
                              problem.inputs.X.row(y_i).transposeROI(), y0);
      theta_perturbed(0, x_i) += 2 * epsilon;
      problem.inputs.function(theta_perturbed,
                              problem.inputs.X.row(y_i).transposeROI(), y1);
      J(x_i, y_i) = (y1(0, 0) - y0(0, 0)) / (2.0 * epsilon);
    }
  }
}

Matrix
evaluate(const std::function<void(const Matrix &, const Matrix &, Matrix &)>
             function,
         const Matrix &theta, const Matrix X) {
  if (X.height() == 0) {
    return Matrix();
  }
  Matrix y(1, X.height());
  for (size_t y_i = 0; y_i < X.height(); y_i++) {
    Matrix mat_y_i(MatrixROI(0, y_i, 1, 1, &y));
    function(theta, X.row(y_i).transposeROI(), mat_y_i);
  }
  return y;
}
}; // namespace basic_matrix
