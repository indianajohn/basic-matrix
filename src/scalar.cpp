#include "scalar.hpp"

namespace basic_matrix {
double sign(const double &x) {
  if (x >= 0.0) {
    return 1.0;
  } else {
    return -1.0;
  }
}
}; // namespace basic_matrix
