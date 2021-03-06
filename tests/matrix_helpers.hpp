#include "matrix.hpp"
#include "test_helpers.hpp"
#include <exception>
#include <iostream>
#include <random>
#include <string>

const static size_t g_random_seed = 2935;
static std::mt19937 gen(g_random_seed);

bool matrixNear(const basic_matrix::Matrix &mat_result,
                const basic_matrix::Matrix &mat_expected,
                const double &tol = 1e-9);

// Macros for line numbers.
#define ASSERT_MATRIX_NEAR_TOL(result, expected, tol)                          \
  ASSERT(matrixNear(result, expected, tol));

#define ASSERT_MATRIX_NEAR(result, expected)                                   \
  ASSERT_MATRIX_NEAR_TOL(result, expected, 1e-9)

void assertContains(const basic_matrix::Matrix &mat_result,
                    const double &target, const double &tol = 1e-9);

double randomDouble(const double &min, const double &max);

template <typename IntType>
double randomInt(const IntType &min, const IntType &max) {
  std::uniform_int_distribution<IntType> dist(min, max);
  return dist(gen);
}

basic_matrix::Matrix randomMatrix(const size_t &width, const size_t &height,
                                  const double &min, const double &max,
                                  const double &min_absolute_value = 0.1);

basic_matrix::Matrix generateNonsingularMatrix(const size_t &width,
                                               const size_t &height);
