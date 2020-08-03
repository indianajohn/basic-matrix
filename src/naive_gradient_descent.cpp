#include "gauss_newton.hpp"
#include "gaussian_elimination.hpp"
#include <iostream>

namespace basic_matrix {

void naiveGradientDescent(OptimizationProblem &problem,
                          const double &initial_alpha) {
  double alpha = initial_alpha;
  Matrix J;
  if (!problem.config.use_initial_condition ||
      problem.outputs.theta.height() != problem.inputs.num_params) {
    problem.outputs.theta = Matrix(1, problem.inputs.num_params);
    for (size_t i = 0; i < problem.inputs.num_params; i++) {
      problem.outputs.theta(0, i) = 1.0;
    }
  }
  for (size_t step = 0; step < problem.config.max_iterations; step++) {
    if (problem.inputs.jacobian.has_value()) {
      problem.inputs.jacobian.value()(problem.outputs.theta, problem.inputs.X,
                                      problem.inputs.y, J);
    } else {
      estimateJacobian(problem, J);
    }
    double cost = problem.inputs.cost_function.value()(
        problem.outputs.theta, problem.inputs.X, problem.inputs.y);
    // Search along a line in Gradient direction, adjust alpha and theta.
    alpha = 0.25 * alpha;
    for (double i = 2.0; i < 10.0; i += 0.5) {
      double new_alpha = 2.0 * i * alpha;
      Matrix theta = problem.outputs.theta - J.transposeROI() * new_alpha;
      double this_cost = problem.inputs.cost_function.value()(
          theta, problem.inputs.X, problem.inputs.y);
      if (this_cost < cost) {
        alpha = new_alpha;
        problem.outputs.theta = theta;
      }
    }
    problem.outputs.num_iterations++;
  }
} // namespace basic_matrix
}; // namespace basic_matrix
