#include "eigenvalues.hpp"
#include "gaussian_elimination.hpp"
#include "matrix.hpp"
#include "matrix_helpers.hpp"
#include <iostream>

using namespace basic_matrix;

void testEigenvalues() {
  {
    Matrix A = {{1, 1}, {0, 2}};
    double tol = 1e-6;
    auto result = eigenvalues(A, tol);
    ASSERT(result.ok());
    assertContains(result, 1.0, tol);
    assertContains(result, 2.0, tol);
  }
  {
    Matrix A = {{3, -1}, {-1, 3}};
    double tol = 1e-6;
    auto result = eigenvalues(A, tol);
    ASSERT(result.ok());
    assertContains(result, 2.0, tol);
    assertContains(result, 4.0, tol);
  }
  // Repeated eigenvalues.
  {
    Matrix A = {{-1, 1.5}, {-1.0 / 6.0, -2.0}};
    double tol = 1e-6;
    auto result = eigenvalues(A, tol);
    // ASSERT(result.ok());
  }
  {
    Matrix A = {{3, 1, 1}, {0, 2, 0}, {1, 1, 3}};
    double tol = 1e-6;
    auto result = eigenvalues(A, tol);
    ASSERT(result.ok());
  }
  // Zero eigenvalues.
  {
    Matrix A = {{-1, 1.5}, {-1.0 / 6.0, -2.0}};
    double tol = 1e-6;
    auto result = eigenvalues(A, tol);
    ASSERT(result.ok());
  }
  {
    Matrix A = {{11.604381, -7.344107}, {0.187676, 10.981659}};
    // this will not have real-valued Eigenvales, and will
    // fail to converge.
    auto result = eigenvalues(A, 1e-6);
    ASSERT_EQ(result.ok(), false);
  }
}

int main(int argc, char **argv) { testEigenvalues(); }
