#include "eigenvalues.hpp"
#include "gaussian_elimination.hpp"
#include "matrix.hpp"
#include "matrix_helpers.hpp"
#include "qr_factorization.hpp"
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
    double tol = 1e-4;
    auto result = eigenvalues(A, tol);
    ASSERT(result.ok());
    assertContains(result, -1.5, 0.1);
  }
  {
    Matrix A = {{3, 1, 1}, {0, 2, 0}, {1, 1, 3}};
    double tol = 1e-6;
    auto result = eigenvalues(A, tol);
    ASSERT(result.ok());
    assertContains(result, 4.0);
    assertContains(result, 2.0);
  }
  {
    Matrix A = {{11.604381, -7.344107}, {0.187676, 10.981659}};
    // this will not have real-valued Eigenvales, and will
    // fail to converge.
    auto result = eigenvalues(A, 1e-6);
    ASSERT_EQ(result.ok(), false);
  }
  {
    double not_converged = 0;
    double total = 0;
    for (size_t trial = 0; trial < 100; trial++) {
      // Generate a non-diagonal matrix with real eigenvalues, by
      // generating a diagonal matrix with random elements and
      // multiplying it by a similarity transform.
      size_t h = 2 + (rand() % 10);
      Matrix D(h, h);
      for (size_t i = 0; i < h; i++) {
        D(i, i) = randomDouble(-0.5, 0.5);
      }
      Matrix R = randomMatrix(h, h, -5.0, 5.0);
      Matrix Q;
      qrFactorize(Q, R);
      // this will not have real-valued Eigenvales, and will
      // fail to converge.
      Matrix A = Q * D * Q.transposeROI() * 100;
      auto result = eigenvalues(A, 1e-5);
      if (!result.ok()) {
        not_converged++;
      }
      total++;
    }
    ASSERT(not_converged / total < 0.05);
  }
}

int main(int argc, char **argv) { testEigenvalues(); }
