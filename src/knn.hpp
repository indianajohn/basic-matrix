#pragma once
#include "matrix.hpp"

namespace basic_matrix {
class KNNClassifier {
public:
  KNNClassifier(const Matrix &X_train, const Matrix &y_train);

  /// classify rows of test_data, returning the class
  /// for each test data row on the corresponding row
  /// in the return value.
  Matrix classify(const Matrix &test_data);

private:
  Matrix m_X_train;
  Matrix m_y_train;
};
}; // namespace basic_matrix
