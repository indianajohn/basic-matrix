#include "eigenvalues.hpp"
#include "matrix.hpp"
#include "matrix_helpers.hpp"
#include <iostream>

using namespace basic_matrix;

void testEigenvalues() {
  {
    Matrix A = {{1, 1}, {0, 2}};
    double tol = 1e-6;
    Matrix result = eigenvalues(A, tol);
    assertContains(result, 1.0, tol);
    assertContains(result, 2.0, tol);
  }
  {
    Matrix A = {{3, -1}, {-1, 3}};
    double tol = 1e-6;
    Matrix result = eigenvalues(A, tol);
    assertContains(result, 2.0, tol);
    assertContains(result, 4.0, tol);
  }
}

int main(int argc, char **argv) { testEigenvalues(); }
