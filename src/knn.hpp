#pragma once
#include "matrix.hpp"

namespace basic_matrix {
class KNNClassifier {
  KNNClassifier(const Matrix &train_data);

  /// classify rows of test_data, returning the class
  /// for each test data row on the corresponding row
  /// in the return value.
  Matrix classify(const Matrix &test_data);

private:
  Matrix m_train_data;
};
}; // namespace basic_matrix
