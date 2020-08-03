#include "knn.hpp"
#include "io.hpp"
#include "matrix.hpp"
#include "test_helpers.hpp"
#include <iostream>

using namespace basic_matrix;

void knnWorks() {
  Matrix X_test = loadFromFile("tests/knn_X_test.csv");
  ASSERT_EQ(X_test.width(), 3072);
  ASSERT_EQ(X_test.height(), 500);
  Matrix X_train = loadFromFile("tests/knn_X_train.csv");
  ASSERT_EQ(X_test.width(), 3072);
  ASSERT_EQ(X_test.height(), 500);
  Matrix y_test = loadFromFile("tests/knn_y_test.csv");
  ASSERT_EQ(y_test.width(), 1);
  ASSERT_EQ(y_test.height(), 500);
  Matrix y_train = loadFromFile("tests/knn_y_train.csv");
  ASSERT_EQ(y_train.width(), 1);
  ASSERT_EQ(y_train.height(), 5000);
  KNNClassifier classifier(X_train, y_train);
  Matrix y_result = classifier.classify(X_test);
}

int main(int argc, char **argv) { knnWorks(); }
