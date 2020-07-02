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

/// Write a matrix to a string suitable for file
/// serialization.
std::string writeToString(const Matrix &mat);

/// Write a matrix to a file.
void writeToFile(const std::string &path, const Matrix &mat);
}; // namespace basic_matrix
