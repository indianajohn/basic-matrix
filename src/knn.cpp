#include "knn.hpp"
#include "standard_functions.hpp"
#include <iostream>
#include <map>

namespace basic_matrix {
KNNClassifier::KNNClassifier(const Matrix &X_train, const Matrix &y_train)
    : m_X_train(X_train), m_y_train(y_train) {}

Matrix KNNClassifier::classify(const Matrix &X, const size_t &k) {
  // Calculate all distances
  Matrix result(1, X.height());
  Matrix XXtt = -2.0 * X * this->m_X_train.transpose();
  Matrix X2 = pow(X, 2);
  Matrix X2_colSum = X2.sumCols();
  Matrix X_train_2 = pow(this->m_X_train, 2);
  Matrix X_train_2_colSum = X_train_2.sumCols();
  X_train_2_colSum.reshape(X_train_2_colSum.height(), X_train_2_colSum.width());
  Matrix twoSum = XXtt + X_train_2_colSum;
  Matrix dist = twoSum + X2_colSum;

  // Rank by nearest
  for (size_t i_test = 0; i_test < dist.height(); i_test++) {
    std::map<double, std::vector<int>> distance_to_class;
    for (size_t i_train = 0; i_train < dist.width(); i_train++) {
      int class_label = this->m_y_train(0, i_train);
      distance_to_class[dist(i_train, i_test)].push_back(class_label);
    }
    // Build class frequncy map
    std::unordered_map<int, size_t> class_to_frequency;

    // Iterate from smallest to largest distance, build frequency table among
    // the set of k nearest matches.
    size_t count = 0;
    for (const auto &pair : distance_to_class) {
      for (const auto &class_label : pair.second) {
        class_to_frequency[class_label]++;
        count++;
        if (count >= k) {
          break;
        }
      }
      if (count >= k) {
        break;
      }
    }
    // Invert frequency to idx, store in an ordered map.
    std::map<size_t, int> frequency_to_class;
    for (const auto &pair : class_to_frequency) {
      size_t class_label = pair.first;
      size_t frequency = pair.second;
      frequency_to_class[frequency] = class_label;
    }
    // Pick the most common class (highest frequency)
    int frequency = frequency_to_class.rbegin()->first;
    int class_label = frequency_to_class.rbegin()->second;
    if (frequency < 2) {
      result(0, i_test) = distance_to_class.begin()->second.front();
    } else {
      result(0, i_test) = class_label;
    }
  }

  return result;
}
}; // namespace basic_matrix
