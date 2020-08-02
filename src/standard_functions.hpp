#pragma once
#include "matrix.hpp"

namespace basic_matrix {
// This header contains functionst that have the basic signature
// M = f(M)

// We define all functions with this macro when possible, so the user has access
// to the most convenient variant: either do the operation in-place,
// save it to a pre-allocated matrix, or make and return a copy.
#define DEFINE_UTIL_FUNC(name)                                                 \
  void in_place_##name(Matrix &output);                                        \
  Matrix name(const Matrix &input);

/// Exponentiation.
DEFINE_UTIL_FUNC(exp);
/// Log
DEFINE_UTIL_FUNC(log);
/// The sigmoid function.
DEFINE_UTIL_FUNC(sigmoid);

struct LogisticRegressionObjective {
  /// The logistic regression energy function.
  /// @param theta - model parameters.
  /// @param X - input.
  /// @param y - output data.
  /// @param E_out - output energy.
  /// @param lambda - regularization weight.
  void energy(const Matrix &theta, const Matrix &X, const Matrix &y,
              Matrix &E_out);
  // Evaluate y for the model.
  void eval(const Matrix &theta, const Matrix &X, Matrix &out_y);

  /// Evaluate the gradient for the model.
  void gradient(const Matrix &theta, const Matrix &X, const Matrix &y,
                Matrix &J);

  /// regularization weight.
  double lambda = 0.0;
};
}; // namespace basic_matrix
