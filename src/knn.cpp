#include "knn.hpp"

namespace basic_matrix {
KNNClassifier::KNNClassifier(const Matrix &X_train, const Matrix &y_train)
    : m_X_train(X_train), m_y_train(y_train) {}

Matrix KNNClassifier::classify(const Matrix &test_data) {
  Matrix result;
  return result;
}
}; // namespace basic_matrix
