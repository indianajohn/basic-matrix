#include "lup_decomposition.hpp"
#include "qr_factorization.hpp"
#include "scalar.hpp"

namespace basic_matrix {
void qrFactorize(Matrix &Q, Matrix &R) {
  Matrix A = R;
  Q = identity(R.height());
  for (size_t k = 0; k < R.width() && k < R.height(); k++) {
    size_t m_k = R.height() - k;
    size_t n_k = R.width() - k;
    // Consider
    // H_k = [ I 0               ]
    //       [ 0 I - 2*v_k*v_k_t ]
    // H_k is a symmetric matrix, as is identity. Therefore,
    // H_k*H_k = H_k*H_k_T=I
    // A = H_1*H_2*...*H_max*H_max*...*H_2*H_1*A
    // = (H_1*H_2*...*H_max)*R
    // where R is diagonal.
    // = Q * R

    // Buid the reflector I - 2*v_k*v_k.transpose()
    Matrix y(MatrixROI(k, k, 1, m_k, &R));
    Matrix e1(y.width(), y.height());
    e1(0, 0) = 1.0;
    Matrix w = y + (sign(y(0, 0)) * y.norm() * e1);
    double factor = 1. / w.norm();
    Matrix v_k = factor * w;
    Matrix A_roi(MatrixROI(k, k, n_k, m_k, &R));

    // Edit R (whose input was A) in place. Subtracting
    // 2*v_k*v_k.transpose() is equvalent to pre-multiplying H_k.
    // We only do it in the region that would be affected
    // by the multiplication to save on flops
    A_roi = A_roi - (2.0 * v_k * (v_k.transposeROI() * A_roi));
    Matrix H_k = identity(Q.height());
    Matrix H_k_roi(MatrixROI(k, k, m_k, m_k, &H_k));
    H_k_roi = H_k_roi - (2.0 * v_k * v_k.transposeROI());

    // Perform the next multiplication for building Q
    Q = Q * H_k;
  }
}
void solveQR(const Matrix &A, Matrix &b) {
  Matrix R = A;
  Matrix Q;
  qrFactorize(Q, R);
  // If Q and R are a QR factorization of A, the least squares solution is equal
  // to: x_hat = (A.transpose()*A).inverse()*A.transpose() * b =
  // R.inverse()*Q.transpose() Therefore: R*x_hat = Q.transpose()*b where
  // Q.transpose()*b is diagonaol. We can therefore solve it by back
  // substitution.
  b = Q.transposeROI() * b;
  solveU(R, b);
}
}; // namespace basic_matrix
