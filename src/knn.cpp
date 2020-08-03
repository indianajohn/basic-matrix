#include "knn.hpp"

namespace basic_matrix {
KNNClassifier::KNNClassifier(const Matrix &train_data)
    : m_train_data(train_data) {}

Matrix KNNClassifier::classify(const Matrix &test_data) {
  Matrix result;
  // TODO
  return result;
}
}; // namespace basic_matrix
