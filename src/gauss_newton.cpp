#include "gauss_newton.hpp"
#include "gaussian_elimination.hpp"
#include <iostream>

namespace basic_matrix {

void gaussNewton(OptimizationProblem &problem) {
  Matrix J;
  if (!problem.config.use_initial_condition ||
      problem.outputs.theta.height() != problem.inputs.num_params) {
    problem.outputs.theta = Matrix(1, problem.inputs.num_params);
    for (size_t i = 0; i < problem.inputs.num_params; i++) {
      problem.outputs.theta(0, i) = 1.0;
    }
  }
  if (problem.inputs.jacobian.has_value()) {
    problem.inputs.jacobian.value()(problem.outputs.theta, problem.inputs.X,
                                    problem.inputs.y, J);
  } else {
    estimateJacobian(problem, J);
  }
  for (size_t step = 0; step < problem.config.max_iterations; step++) {
    problem.outputs.num_iterations++;
    Matrix y = evaluate(problem.inputs.function, problem.outputs.theta,
                        problem.inputs.X);
    Matrix r = problem.inputs.y - y;
    double cost = 0;
    if (problem.inputs.cost_function.has_value()) {
      cost = problem.inputs.cost_function.value()(
          problem.outputs.theta, problem.inputs.X, problem.inputs.y);
    } else {
      cost = r.norm();
    }
    if (cost < problem.config.cost_threshold) {
      return;
    }

    // Gauss-Newton update:
    // (https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm)
    // theta_(k+1) = theta_k + (J_f_t*J_f)^(-1)*J_f_t*r
    Matrix delta = J.transposeROI() * r;
    solveByGaussianElimination(J.transposeROI() * J, delta);
    problem.outputs.theta = problem.outputs.theta + delta;
  }
}
}; // namespace basic_matrix
