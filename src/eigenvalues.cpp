#include "eigenvalues.hpp"
#include "qr_factorization.hpp"
#include <iostream>

namespace basic_matrix {
/// This function currently only uses the QR algorithm. It would be better
/// to have multiple methods that switch based on matrix size.
Matrix eigenvalues(const Matrix &A, const double &convergence_threshold) {
  if (A.height() != A.width()) {
    throw std::runtime_error(
        "Cannot compute eigenvalues for non-square matrix.");
  }
  Matrix RQ = A;
  Matrix eigenvalues(1, RQ.height());
  for (size_t i = 0; i < A.height(); i++) {
    eigenvalues(0, i) = RQ(i, i);
  }

  double error = std::numeric_limits<double>::max();
  while (error > convergence_threshold) {
    Matrix Q;
    // R is stored in RQ
    qrFactorize(Q, RQ);
    // R * Q
    RQ = RQ * Q;
    error = 0;
    for (size_t i = 0; i < Q.height(); i++) {
      error += std::fabs(eigenvalues(0, i) - RQ(i, i));
      eigenvalues(0, i) = RQ(i, i);
    }
  }
  return eigenvalues;
}
}; // namespace basic_matrix
