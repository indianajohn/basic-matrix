#include "standard_functions.hpp"

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

}; // namespace basic_matrix
