#pragma once
#include "matrix.hpp"
#include <string>

namespace basic_matrix {
/// Read a matrix from an Octave/Matlab-compatible
/// CSV string.
Matrix parseFromString(const std::string &str);
/// Read a matrix from an Octave/Matlab-compatible
/// CSV file.
Matrix loadFromFile(const std::string &str);
}; // namespace basic_matrix
