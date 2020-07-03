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
/// The sigmoid function.
DEFINE_UTIL_FUNC(sigmoid);

}; // namespace basic_matrix
