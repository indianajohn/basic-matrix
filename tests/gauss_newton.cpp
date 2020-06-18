#include "gauss_newton.hpp"
#include "matrix_helpers.hpp"

using namespace basic_matrix;

void linearFunction(const Matrix &theta, const Matrix &x, Matrix &y) {
  y = theta.transpose() * x;
}

void nonlinearFunction(const Matrix &theta, const Matrix &x, Matrix &y) {
  double sum = 0;
  for (size_t i = 0; i < x.height(); i++) {
    sum += theta(0, i) * powf(x(0, i), i);
    if (i > 0) {
      sum += theta(0, i - 1) * powf(x(0, i - 1), i);
    }
  }
  y = Matrix(1, 1);
  y(0, 0) = sum;
}

void testLinearSystem() {
  OptimizationProblem problem;
  problem.inputs.num_params = randomInt(1, 10);
  Matrix theta_gt = randomMatrix(1, problem.inputs.num_params, -10.0, 10.0);
  problem.outputs.theta = randomMatrix(1, problem.inputs.num_params, -3.0, 3.0);
  size_t num_samples = problem.inputs.num_params;
  problem.inputs.y = Matrix(1, num_samples);
  for (size_t i = 0; i < num_samples; i++) {
    Matrix row = randomMatrix(problem.inputs.num_params, 1, -10.0, 10.0);
    problem.inputs.x.push_back(row.transpose());
    problem.inputs.y(0, i) = (row * theta_gt)(0, 0);
  }
  problem.inputs.function = &linearFunction;
  gaussNewton(problem);
  assertMatrixNear(problem.outputs.theta, theta_gt, 1e-6);
  ASSERT(problem.outputs.num_iterations > 0);
}

void testNonlinearSystem() {
  OptimizationProblem problem;
  problem.inputs.num_params = randomInt(1, 10);
  Matrix theta_gt = randomMatrix(1, problem.inputs.num_params, -10.0, 10.0);
  problem.outputs.theta = randomMatrix(1, problem.inputs.num_params, -3.0, 3.0);
  size_t num_samples = problem.inputs.num_params;
  problem.inputs.y = Matrix(1, num_samples);
  for (size_t i = 0; i < num_samples; i++) {
    Matrix row = randomMatrix(problem.inputs.num_params, 1, -10.0, 10.0);
    problem.inputs.x.push_back(row.transpose());
    Matrix y;
    nonlinearFunction(theta_gt, problem.inputs.x.back(), y);
    problem.inputs.y(0, i) = y(0, 0);
  }
  problem.inputs.function = &nonlinearFunction;
  gaussNewton(problem);
  assertMatrixNear(problem.outputs.theta, theta_gt, 1e-6);
  ASSERT(problem.outputs.num_iterations > 0);
}

void gaussNewtonWorksForLinearSystem() {
  size_t num_trials = 10;
  for (size_t i = 0; i < num_trials; i++) {
    testLinearSystem();
  }
}

void gaussNewtonWorksForNonlinearSystem() {
  size_t num_trials = 10;
  for (size_t i = 0; i < num_trials; i++) {
    testNonlinearSystem();
  }
}

int main() {
  gaussNewtonWorksForLinearSystem();
  gaussNewtonWorksForNonlinearSystem();
  return 0;
}
