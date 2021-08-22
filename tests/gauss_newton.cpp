#include "gauss_newton.hpp"
#include "io.hpp"
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
  problem.inputs.X = Matrix(problem.inputs.num_params, num_samples);
  for (size_t i = 0; i < num_samples; i++) {
    Matrix row = randomMatrix(problem.inputs.num_params, 1, -10.0, 10.0);
    problem.inputs.X.row(i) = row;
    problem.inputs.y(0, i) = (row * theta_gt)(0, 0);
  }
  problem.inputs.function = &linearFunction;
  gaussNewton(problem);
  ASSERT_MATRIX_NEAR_TOL(problem.outputs.theta, theta_gt, 1e-6);
  ASSERT(problem.outputs.num_iterations > 0);
}

void testNonlinearSystem() {
  OptimizationProblem problem;
  problem.inputs.num_params = randomInt(1, 10);
  Matrix theta_gt = randomMatrix(1, problem.inputs.num_params, -10.0, 10.0);
  problem.outputs.theta = randomMatrix(1, problem.inputs.num_params, -3.0, 3.0);
  size_t num_samples = problem.inputs.num_params;
  problem.inputs.y = Matrix(1, num_samples);
  problem.inputs.X = Matrix(problem.inputs.num_params, num_samples);
  for (size_t i = 0; i < num_samples; i++) {
    Matrix row = randomMatrix(problem.inputs.num_params, 1, -10.0, 10.0);
    problem.inputs.X.row(i) = row;
    Matrix y;
    nonlinearFunction(theta_gt, problem.inputs.X.row(i).transposeROI(), y);
    problem.inputs.y(0, i) = y(0, 0);
  }
  problem.inputs.function = &nonlinearFunction;
  gaussNewton(problem);
  ASSERT_MATRIX_NEAR_TOL(problem.outputs.theta, theta_gt, 1e-6);
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

void gaussNewtonWorksFor1DLinearSystemWithNoise() {
  Matrix Xy = loadFromFile("tests/1d_linear_regression.txt");
  Matrix X(2, Xy.height());
  Matrix x_second_column(MatrixROI(1, 0, 1, Xy.height(), &X));
  x_second_column = Matrix(MatrixROI(0, 0, 1, Xy.height(), &Xy));
  for (size_t i = 0; i < Xy.height(); i++) {
    X(0, i) = 1;
  }
  Matrix y = Matrix(MatrixROI(1, 0, 1, Xy.height(), &Xy));
  Matrix theta_normal_eqn = (X.transpose() * X).inverse() * X.transpose() * y;

  OptimizationProblem problem;
  problem.inputs.num_params = 2;
  problem.outputs.theta = randomMatrix(1, problem.inputs.num_params, -3.0, 3.0);
  problem.inputs.X = X;
  problem.inputs.y = y;
  problem.inputs.function = &linearFunction;
  gaussNewton(problem);
  ASSERT_MATRIX_NEAR_TOL(problem.outputs.theta, theta_normal_eqn, 1e-6);
}

void gaussNewtonWorksFor2DLinearSystemWithNoise() {
  Matrix Xy = loadFromFile("tests/2d_linear_regression.txt");
  Matrix X(3, Xy.height());
  Matrix x_first_two_columns(MatrixROI(1, 0, 2, Xy.height(), &X));
  x_first_two_columns = Matrix(MatrixROI(0, 0, 2, Xy.height(), &Xy));
  for (size_t i = 0; i < Xy.height(); i++) {
    X(0, i) = 1;
  }
  Matrix y = Matrix(MatrixROI(2, 0, 1, Xy.height(), &Xy));
  Matrix theta_normal_eqn = (X.transpose() * X).inverse() * X.transpose() * y;

  OptimizationProblem problem;
  problem.inputs.num_params = 3;
  problem.outputs.theta = randomMatrix(1, problem.inputs.num_params, -3.0, 3.0);
  problem.inputs.X = X;
  problem.inputs.y = y;
  problem.inputs.function = &linearFunction;
  problem.config.max_iterations = 1000;
  gaussNewton(problem);
  ASSERT_MATRIX_NEAR_TOL(problem.outputs.theta, theta_normal_eqn, 10.0);
}

void gaussNewtonWorksForLogisticRegression() {}

int main() {
  gaussNewtonWorksForLinearSystem();
  gaussNewtonWorksForNonlinearSystem();
  gaussNewtonWorksFor1DLinearSystemWithNoise();
  gaussNewtonWorksFor2DLinearSystemWithNoise();
  return 0;
}
