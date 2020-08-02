#include "gauss_newton.hpp"
#include "matrix_helpers.hpp"

using namespace basic_matrix;

void linearFunction(const Matrix &theta, const Matrix &x, Matrix &y) {
  y = theta.transpose() * x;
}

void linear1D() {
  Matrix A = {1};
  OptimizationProblem problem;
  problem.inputs.X = {1};
  problem.outputs.theta = {0};
  problem.inputs.y = A * problem.inputs.X.row(0);
  problem.inputs.function = &linearFunction;
  problem.inputs.num_params = 1;
  Matrix J;
  estimateJacobian(problem, J);
  // J should be d_y/d_theta
  // Analytically the function is y = theta * x
  // So d_y / d_theta / x = 1
  ASSERT_NEAR(J(0, 0), 1.0);
}

void linearND() {
  size_t num_trials = 10;
  for (size_t trial = 0; trial < num_trials; trial++) {
    OptimizationProblem problem;
    problem.inputs.num_params = randomInt(1, 10);
    Matrix theta_gt = randomMatrix(1, problem.inputs.num_params, -10.0, 10.0);
    problem.outputs.theta =
        randomMatrix(1, problem.inputs.num_params, -3.0, 3.0);
    size_t num_samples = problem.inputs.num_params;
    problem.inputs.y = Matrix(1, num_samples);
    problem.inputs.X = Matrix(problem.inputs.num_params, num_samples);
    for (size_t i = 0; i < num_samples; i++) {
      problem.inputs.X.row(i) =
          randomMatrix(problem.inputs.num_params, 1, -10.0, 10.0);
      problem.inputs.y(0, i) = (problem.inputs.X.row(i) * theta_gt)(0, 0);
    }
    problem.inputs.function = &linearFunction;
    Matrix J;
    estimateJacobian(problem, J);

    // Check that the linearized system is equivalent to the linear system
    Matrix u_0 = problem.outputs.theta;
    Matrix u = randomMatrix(1, problem.inputs.num_params, -5.0, 5.0);
    Matrix y_u_0 = evaluate(&linearFunction, u_0, problem.inputs.X);
    Matrix y_val = evaluate(&linearFunction, u, problem.inputs.X);
    Matrix result = y_u_0 + J * (u - u_0);
    ASSERT_MATRIX_NEAR_TOL(result, y_val, 1e-3);
  }
}

int main() {
  linear1D();
  linearND();
  return 0;
}
