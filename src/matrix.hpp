#pragma once
#include <ostream>
#include <random>
#include <stddef.h>
#include <unordered_map>
#include <vector>

namespace basic_matrix {
class Matrix;

class BoundingBox {
public:
  BoundingBox(const size_t &min_x_in = std::numeric_limits<size_t>::max(),
              const size_t &min_y_in = std::numeric_limits<size_t>::max(),
              const size_t &max_x_in = 0, const size_t &max_y_in = 0)
      : m_min_x(min_x_in), m_min_y(min_y_in), m_max_x(max_x_in),
        m_max_y(max_y_in) {}
  std::vector<std::pair<size_t, size_t>> points() const;
  void extend(const std::pair<size_t, size_t> point);
  bool isInside(const size_t &x, const size_t &y) const;
  const size_t minX() const;
  const size_t minY() const;
  const size_t maxX() const;
  const size_t maxY() const;

private:
  size_t m_min_x;
  size_t m_min_y;
  size_t m_max_x;
  size_t m_max_y;
};

/// A Matrix ROI.
///
/// By example:
/// [  1,  2,  3,  4, 5  ]
/// [  6,  7,  8,  9, 10 ]
/// [ 11, 12, 13, 14, 15 ]
/// [ 16, 17, 18, 19, 20 ]
///
/// The ROI [1, 1, 2, 2, 0, 0] is:
/// [  7  8 ]
/// [ 12 13 ]
///
/// The ROI [1, 1, 2, 2, 1, 1] is:
/// [ ?  ?  ? ]
/// [ ?  7  8 ]
/// [ ? 12 13 ]
///
/// Transpose:
/// The ROI [1, 1, 3, 2, 1, 1] is:
/// [ ?  ?  ? ]
/// [ ?  7 12 ]
/// [ ?  8 13 ]
/// [ ?  9 14 ]
///
/// It is the caller's responsibility to ensure that the entire
/// indexing space of the ROI matrix points to an ROI. If not,
/// access out of bounds of pre-defined ROIs will result in bounds
/// errors.
///
/// ROIs must not outlive the matrix to which they point.
//
class MatrixROI {
public:
  MatrixROI(const size_t &src_x_in, const size_t &src_y_in,
            const size_t &width_in, const size_t &height_in, Matrix *matrix_in,
            const size_t &dst_x_in = 0, const size_t &dst_y_in = 0,
            const bool &transposed_in = false)
      : m_src_x(src_x_in), m_src_y(src_y_in), m_width(width_in),
        m_height(height_in), m_matrix(matrix_in), m_dst_x(dst_x_in),
        m_dst_y(dst_y_in), m_transposed(transposed_in) {
    m_src_bounding_box = calculateSrcBoundingBox();
    m_dst_bounding_box = calculateDstBoundingBox();
  }
  bool operator==(const MatrixROI &other) const {
    return other.srcX() == srcX() && other.srcY() == srcY() &&
           other.width() == width() && other.height() == height() &&
           other.dstX() == dstX() && other.dstY() == dstY() &&
           other.transposed() == transposed();
  };

  /// Checks that the bounds of the contained matrix are obeyed by the bounding
  /// box
  bool isConsistent() const;
  /// Transformation
  void srcToDst(const size_t &x, const size_t &y, size_t &out_x,
                size_t &out_y) const;
  /// Transformation
  void dstToSrc(const size_t &x, const size_t &y, size_t &out_x,
                size_t &out_y) const;

  BoundingBox srcBoundingBox() const;
  BoundingBox dstBoundingBox() const;

  size_t srcX() const;
  size_t srcY() const;
  size_t dstX() const;
  size_t dstY() const;
  size_t width() const;
  size_t height() const;
  size_t dstWidth() const;
  size_t dstHeight() const;
  bool transposed() const;
  /// Checks whether a point is inside the ROI. x & y are in the
  /// destination coordinate system
  bool isInside(const size_t &x, const size_t &y) const;

  /// Checks whether there is an overlap region between bounding boxes.
  bool overlaps(const MatrixROI &roi) const;
  Matrix *matrix() const;

private:
  /// The elements in the source matrix that correspond to 0,0 in ROI space.
  size_t m_src_x;
  size_t m_src_y;

  /// The size of the ROI in the source matrix space (pre-transpose)
  size_t m_width;
  size_t m_height;

  /// The matrix to which we point
  Matrix *m_matrix;

  /// The region in ROI space this ROI occupies.
  size_t m_dst_x;
  size_t m_dst_y;

  /// Whether or not to transpose
  bool m_transposed;

  BoundingBox calculateSrcBoundingBox() const;
  BoundingBox calculateDstBoundingBox() const;

  BoundingBox m_src_bounding_box;
  BoundingBox m_dst_bounding_box;
};
}; // namespace basic_matrix

namespace std {
template <> struct hash<basic_matrix::MatrixROI> {
public:
  size_t operator()(const basic_matrix::MatrixROI &roi) const {
    return std::hash<size_t>()(roi.srcX()) ^ std::hash<size_t>()(roi.srcY()) ^
           std::hash<size_t>()(roi.dstX()) ^ std::hash<size_t>()(roi.dstY()) ^
           std::hash<bool>()(roi.transposed()) ^
           std::hash<size_t>()(roi.width()) ^ std::hash<size_t>()(roi.height());
  }
};
}; // namespace std
namespace basic_matrix {

class Matrix {
public:
  /// Initialize a 0x0 matrix. For the purpose of construcitng matrices
  /// that are to be modified in-place. The result of this constructor
  /// is not ok() and can be used to return failure.
  Matrix();
  Matrix(const Matrix &other);
  Matrix(const size_t &width, const size_t &height);

  Matrix(const std::vector<std::vector<double>> &input);
  Matrix(const std::initializer_list<std::initializer_list<double>> &input);
  Matrix(const std::vector<double> &input);
  Matrix(const std::initializer_list<double> &input);
  // Wrap another matrix
  Matrix(MatrixROI roi);
  Matrix(std::vector<MatrixROI> &rois);

  size_t width() const;
  size_t height() const;
  const double &operator()(const size_t &x, const size_t &y) const;
  double &operator()(const size_t &x, const size_t &y);
  Matrix &operator=(const Matrix &mat);
  Matrix transpose() const;

  /// Leaving storage intact, change the width and height of
  /// a matrix. The product of width and height must be the
  /// same after the reshape operation.
  void reshape(const size_t &new_width, const size_t &new_height);

  /// Sum all rows and save the result in a single column matrix.
  Matrix sumRows() const;

  /// Sum all columns and save the result in a single row matrix.
  Matrix sumCols() const;

  /// An ROI that points to the original matrix storage
  /// but is transposed compared to the original.
  Matrix transposeROI();
  const Matrix transposeROI() const;

  /// Accessors for row/column ROIs
  Matrix row(const size_t &v);
  const Matrix row(const size_t &v) const;
  Matrix col(const size_t &v);
  const Matrix col(const size_t &v) const;

  double norm() const;

  // Operators
  const Matrix operator+(const Matrix &other) const;
  const Matrix operator-(const Matrix &other) const;
  const Matrix operator*(const Matrix &other) const;
  const Matrix operator+(const double &scalar) const;
  const Matrix operator-(const double &scalar) const;
  const Matrix operator*(const double &scalar) const;
  const Matrix operator/(const double &scalar) const;
  const Matrix operator-() const;

  // In-place operators

  /// In-place addition; adds a scalar to all elements.
  /// Equivalent to:
  /// A = scalar + A = A + scalar
  void operator+=(const double &scalar);
  /// In-place subtraction. This matrix will be changed
  /// to the value of A = A - scalar. If you want to perform
  /// A = scalar - A in-place, you could use:
  /// A *= -scalar;
  /// A += scalar;
  void operator-=(const double &scalar);
  /// in-place multiplication. Equivalent to:
  /// A = scalar * A = A * scalar
  void operator*=(const double &scalar);
  /// In-place division. Equivalent to:
  /// A = A / scalar;
  void operator/=(const double &scalar);
  /// In-place multiplication. Equivalent to:
  /// A = A + matrix
  void operator+=(const Matrix &matrix);
  /// In-place subtraction. Equivalent to:
  /// A = A - matrix
  void operator-=(const Matrix &matrix);
  // Since matrix multiplication results in a matrix
  // with diffrent storage dimensions, a reallocation
  // will happen anyway, so *= doesn't make sense.

  /// Returns the matrix [this other]
  /// Requires that this and other are of equal height.
  const Matrix concatRight(const Matrix &other) const;

  /// Returns the matrix [this other]
  /// Requires that this and other are of equal width.
  const Matrix concatDown(const Matrix &other) const;

  /// Implementation of the inner product
  const Matrix dot(const Matrix &other) const;

  /// Return the inverse of a square matrix. Returns a matrix
  /// that is not ok() if the matrix is singular.
  Matrix inverse() const;

  void swapRows(const size_t i, const size_t j);
  void swapCols(const size_t i, const size_t j);

  /// Matrix determinant
  double det() const;

  /// Whether or not this matrix is valid. This method is
  /// is used to implement the optional return pattern
  /// for operations for which a result is not assured - for
  /// example, matrix inversion.
  bool ok() const;

  // Wrap another matrix.
  void addROI(MatrixROI roi_to_add);

  const double *data() const { return &m_storage[0]; }

  /// A pointer to the storage. This is only guaranteed to
  /// actually point to the values of the matrix if the
  /// matrix storage is contiguous.
  double *data();

  /// Is the matrix storage contiguous?
  bool contiguous() const;

private:
  size_t getIndex(const size_t &x, const size_t &y) const;
  void init(const std::vector<std::vector<double>> &input);
  size_t m_width;
  size_t m_height;
  std::vector<double> m_storage;
  bool m_ok = true;
  std::vector<MatrixROI> m_rois;
};

std::ostream &operator<<(std::ostream &os, const Matrix &dt);
std::ostream &operator<<(std::ostream &os, const MatrixROI &roi);
std::ostream &operator<<(std::ostream &os, const BoundingBox &bb);

Matrix identity(const size_t &size);

/// Compute determinant based on definition.
double bruteForceDeterminant(const Matrix &mat);

/// Multiply without any vectorization.
void naiveMultiply(const Matrix &A, const Matrix &B, Matrix &C);

/// Multiplication with CPU vectorization.
void simdMultiply(const Matrix &A, const Matrix &B, Matrix &C);

basic_matrix::Matrix operator*(const double &a, const basic_matrix::Matrix &b);
basic_matrix::Matrix operator+(const double &a, const basic_matrix::Matrix &b);
basic_matrix::Matrix operator-(const double &a, const basic_matrix::Matrix &b);
basic_matrix::Matrix operator/(const double &a, const basic_matrix::Matrix &b);

} // namespace basic_matrix

#define PRINT_MATRIX(VAR)                                                      \
  std::cout << #VAR "=" << std::endl << VAR << std::endl;
