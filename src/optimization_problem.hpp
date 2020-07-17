#pragma once

#include "matrix.hpp"
#include <functional>
#include <optional>

namespace basic_matrix {
/// A nonlinear least squares problem.
struct OptimizationProblem {
  struct Inputs {
    /// Assume there exits a nonlinear function of the form:
    /// y = f(x, theta)
    /// where
    /// * y is a column vector representing the output of the function.
    /// * x is a column vector representing the input of the function.
    /// * theta is a collection of model parameters.
    /// The target function should implement that function in the form:
    ///
    /// void targetFunction(const Matrix& theta, const Matrix& x, Matrix& y);
    ///
    /// The function uses theta and x to calculate y. If y is already a
    /// x.width() x 1 matrix, then the storage is reused. Otherwise the
    /// matrix is discarded and recreated and new storage is initialized.
    std::function<void(const Matrix &, const Matrix &, Matrix &)> function;

    /// A cost function. If none is given, ||y - f(x)|| is used. The signature
    /// is:
    /// double targetFunction(const Matrix& theta, const std::vector<Matrix>& x,
    /// const Matrix& y);
    std::optional<std::function<double(
        const Matrix &, const std::vector<Matrix> &, const Matrix &)>>
        cost_function;

    /// An analytical Jacobian. If not provided, the Jacobian will be estimated
    /// numerically. The format is:
    /// double targetFunction(const Matrix& theta, const Matrix& x, const
    /// Matrix& y, Matrix& J);
    std::optional<std::function<void(
        const Matrix &, const std::vector<Matrix> &, const Matrix &, Matrix &)>>
        jacobian;

    /// The function inputs
    std::vector<Matrix> x;

    /// The function outputs
    Matrix y;

    /// Number of parameters in target function.
    size_t num_params;
  };

  struct Configuration {
    /// Whether or not to regard the contents of theta as an initial condition.
    bool use_initial_condition = true;

    /// The maximum iterations to attempt before giving up.
    size_t max_iterations = 100;

    /// The cost at which to stop working.
    double cost_threshold = 1e-7;
  };

  /// Outputs of the optimization algorithm.
  struct Outputs {

    /// The model parameters, to be written by the optimization algorithm.
    /// If use_initial_condition is true, these are used as the initial state
    /// of the optimization.
    Matrix theta;

    /// The final vector cost, to be written by the optimization algorithm.
    Matrix vector_cost;

    /// The final scalar cost, to be written by the optimization algorithm.
    double scalar_cost = std::numeric_limits<double>::max();

    /// The number of steps it took to reach a solution.
    size_t num_iterations = 0;

    /// Whether or not the cost threshold was reached.
    bool converged = false;
  };

  Inputs inputs;
  Configuration config;
  Outputs outputs;
};

/// Estimate the Jacobian of the function about x by perturbing it and
/// calculating the numerical derivatives.
/// A Jacobian is a matrix of the form:
///
/// [ d_y_0_x_0 d_y_0_x_1 ... d_y_0_x_N ]
/// [ d_y_1_x_0 d_y_1_x_1 ... d_y_0_x_N ]
/// [ ...                               ]
/// [ d_y_M_x_0 d_y_1_x_1 ... d_y_M_x_N ]
///
/// Where N is the number of inputs to the function to linearize,
/// and M is the nmber of outputs.
void estimateJacobian(const OptimizationProblem &problem, Matrix &J,
                      const double epsilon = 1e-7);

/// Calculate y
Matrix
evaluate(const std::function<void(const Matrix &, const Matrix &, Matrix &)>
             function,
         const Matrix &theta, const std::vector<Matrix> x);
}; // namespace basic_matrix
