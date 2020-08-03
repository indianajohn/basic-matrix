#include "knn.hpp"
#include "standard_functions.hpp"
#include <iostream>

namespace basic_matrix {
KNNClassifier::KNNClassifier(const Matrix &X_train, const Matrix &y_train)
    : m_X_train(X_train), m_y_train(y_train) {}

Matrix KNNClassifier::classify(const Matrix &X) {
  Matrix result(1, X.height());
  Matrix XXtt = -2.0 * X * this->m_X_train.transpose();
  Matrix X2 = pow(X, 2);
  Matrix X2_colSum = X2.sumCols();
  Matrix X_train_2 = pow(this->m_X_train, 2);
  Matrix X_train_2_colSum = X_train_2.sumCols();
  X_train_2_colSum.reshape(X_train_2_colSum.height(), X_train_2_colSum.width());
  Matrix twoSum = XXtt + X_train_2_colSum;
  Matrix dist = twoSum + X2_colSum;
  // TODO: knn where k != 1
  for (size_t i_test = 0; i_test < dist.height(); i_test++) {
    double min_val = std::numeric_limits<double>::max();
    size_t min_training_idx = 0;
    for (size_t i_train = 0; i_train < dist.width(); i_train++) {
      if (dist(i_train, i_test) < min_val) {
        min_val = dist(i_train, i_test);
        min_training_idx = i_train;
      }
    }
    result(0, i_test) = this->m_y_train(0, min_training_idx);
  }
  return result;
}
}; // namespace basic_matrix
