#include "eigenvalues.hpp"
#include "qr_factorization.hpp"
#include "scalar.hpp"
#include <iostream>

namespace basic_matrix {
namespace {
void hessenbergReduction(Matrix &A) {
  int n = A.height();
  if (n > 2) {
    Matrix a1(MatrixROI(0, 1, 1, A.height() - 1, &A));
    Matrix a1_reflected(MatrixROI(1, 0, A.width() - 1, 1, &A));
    Matrix a11(MatrixROI(1, 1, A.width() - 1, A.height() - 1, &A));
    Matrix e1(1, n - 1);
    auto sgn = sign(a1(0, 0));
    e1(0, 0) = 1;
    Matrix v = (a1 + sgn * a1.norm() * e1);
    v /= v.norm();
    Matrix Q1 = identity(n - 1) - 2 * (v * v.transpose());
    a1 = Q1 * a1;
    a1_reflected = (Q1 * a1_reflected.transpose()).transpose();
    a11 = Q1 * a11 * Q1.transposeROI();
    if (n > 1) {
      hessenbergReduction(a11);
    }
  }
}
}; // namespace

/// This function currently only uses the QR algorithm. It would be better
/// to have multiple methods that switch based on matrix size.
Matrix eigenvalues(const Matrix &A, const double &convergence_threshold,
                   const size_t &max_iterations) {
  if (A.height() != A.width()) {
    throw std::runtime_error(
        "Cannot compute eigenvalues for non-square matrix.");
  }
  // Make A upper diagonal while retaining eigenvalues.
  Matrix RQ = A;
  Matrix Q;

  // Initialize.
  Matrix eigenvalues(1, RQ.height());

  for (int n = RQ.height() - 1; n >= 0; n--) {
    Matrix eye = identity(n + 1);
    Matrix A_focus(MatrixROI(0, 0, n + 1, n + 1, &RQ));
    hessenbergReduction(A_focus);
    double num_iter = 0;
    while (fabs(RQ(n - 1, n)) > convergence_threshold &&
           num_iter < max_iterations) {
      if (n > 0) {
        // Calculate Wilkinson shift
        double a_minus = A_focus(n - 1, n - 1);
        double a = A_focus(n, n);
        double b = A_focus(n, n - 1);
        double delta = (a_minus - a) / 2.0;
        double mu =
            a -
            pow(b, 2) / (delta + sign(delta) * sqrt(pow(delta, 2) + pow(b, 2)));
        if (delta < 1e-5) {
          mu = a - fabs(b);
        }
        if (A_focus.height() < 3) {
          mu = 0;
        }
        Matrix Q;
        A_focus -= mu * eye;
        qrFactorize(Q, A_focus);
        A_focus = A_focus * Q + mu * eye;
      }
      num_iter++;
    }
    if (num_iter >= max_iterations) {
      return Matrix();
    }
    eigenvalues(0, n) = RQ(n, n);
  }
  return eigenvalues;
}
}; // namespace basic_matrix
