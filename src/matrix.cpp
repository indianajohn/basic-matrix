#include "matrix.hpp"
#include "lup_decomposition.hpp"
#include <cassert>
#include <iomanip>
#include <iostream>

namespace basic_matrix {

std::vector<std::pair<size_t, size_t>> BoundingBox::points() const {
  std::vector<std::pair<size_t, size_t>> points;
  for (size_t x : {m_min_x, m_max_x}) {
    for (size_t y : {m_min_y, m_max_y}) {
      points.push_back(std::make_pair(x, y));
    }
  }
  return points;
}
void BoundingBox::extend(const std::pair<size_t, size_t> point) {
  if (point.first > m_max_x)
    m_max_x = point.first;
  if (point.first < m_min_x)
    m_min_x = point.first;
  if (point.second > m_max_y)
    m_max_y = point.second;
  if (point.second < m_min_y)
    m_min_y = point.second;
}
bool BoundingBox::isInside(const size_t &x, const size_t &y) const {
  bool x_inside = x >= m_min_x && x < m_max_x;
  bool y_inside = y >= m_min_y && y < m_max_y;
  return x_inside && y_inside;
}

BoundingBox MatrixROI::calculateSrcBoundingBox() const {
  return BoundingBox(m_src_x, m_src_y, m_src_x + m_width, m_src_y + m_height);
}
BoundingBox MatrixROI::srcBoundingBox() const { return m_src_bounding_box; }

BoundingBox MatrixROI::calculateDstBoundingBox() const {
  BoundingBox src_bb = srcBoundingBox();
  BoundingBox dst_bb;
  std::vector<std::pair<size_t, size_t>> points = src_bb.points();
  for (std::pair<size_t, size_t> src_point : points) {
    std::pair<size_t, size_t> dst_point;
    srcToDst(src_point.first, src_point.second, dst_point.first,
             dst_point.second);
    dst_bb.extend(dst_point);
  }
  return dst_bb;
}
BoundingBox MatrixROI::dstBoundingBox() const { return m_dst_bounding_box; }

const size_t BoundingBox::minX() const { return m_min_x; }
const size_t BoundingBox::minY() const { return m_min_y; }
const size_t BoundingBox::maxX() const { return m_max_x; }
const size_t BoundingBox::maxY() const { return m_max_y; }

size_t MatrixROI::srcX() const { return m_src_x; }
size_t MatrixROI::srcY() const { return m_src_y; }
size_t MatrixROI::dstX() const { return m_dst_x; }
size_t MatrixROI::dstY() const { return m_dst_y; }
size_t MatrixROI::width() const { return m_width; }
size_t MatrixROI::height() const { return m_height; }
size_t MatrixROI::dstWidth() const {
  if (transposed()) {
    return m_height;
  } else {
    return m_width;
  }
}
size_t MatrixROI::dstHeight() const {
  if (transposed()) {
    return m_width;
  } else {
    return m_height;
  }
}

bool MatrixROI::transposed() const { return m_transposed; }

Matrix *MatrixROI::matrix() const { return m_matrix; }

bool MatrixROI::isInside(const size_t &x, const size_t &y) const {
  BoundingBox dst_bb = dstBoundingBox();
  return dst_bb.isInside(x, y);
}

bool MatrixROI::overlaps(const MatrixROI &roi) const {
  BoundingBox b1 = roi.dstBoundingBox();
  BoundingBox b2 = this->dstBoundingBox();
  for (auto point : b1.points()) {
    if (b2.isInside(point.first, point.second)) {
      return true;
    }
  }
  return false;
}

bool MatrixROI::isConsistent() const {
  std::vector<std::pair<size_t, size_t>> points = srcBoundingBox().points();
  for (auto point : points) {
    if (point.first > m_matrix->width())
      return true;
    if (point.second > m_matrix->height())
      return true;
  }
  return false;
}
void MatrixROI::srcToDst(const size_t &x, const size_t &y, size_t &out_x,
                         size_t &out_y) const {
  if (x < this->m_src_x) {
    throw std::out_of_range(std::to_string(x) + "," +
                            std::to_string(this->m_src_x));
  }
  if (y < this->m_src_y) {
    throw std::out_of_range(std::to_string(y) + "," +
                            std::to_string(this->m_src_y));
  }
  // start out in the same place as src
  out_x = x;
  out_y = y;

  // apply src adjustment
  out_x -= m_src_x;
  out_y -= m_src_y;

  // tranpose if needed
  if (this->transposed()) {
    size_t y_tmp = out_y;
    out_y = out_x;
    out_x = y_tmp;
  }
  // Apply dst adjustment
  out_x += m_dst_x;
  out_y += m_dst_y;
}
void MatrixROI::dstToSrc(const size_t &x, const size_t &y, size_t &out_x,
                         size_t &out_y) const {
  out_x = x;
  out_y = y;

  out_x -= m_dst_x;
  out_y -= m_dst_y;

  if (this->transposed()) {
    size_t y_tmp = out_y;
    out_y = out_x;
    out_x = y_tmp;
  }

  out_x += m_src_x;
  out_y += m_src_y;
}

const double &Matrix::operator()(const size_t &x, const size_t &y) const {
  for (auto roi : m_rois) {
    if (roi.isInside(x, y)) {
      size_t new_x = x;
      size_t new_y = y;
      roi.dstToSrc(x, y, new_x, new_y);
      return (*roi.matrix())(new_x, new_y);
    }
  }
  return m_storage[getIndex(x, y)];
}

Matrix::Matrix(const Matrix &other)
    : m_width(other.width()), m_height(other.height()),
      m_storage(other.width() * other.height()) {
  for (size_t x = 0; x < width(); x++) {
    for (size_t y = 0; y < height(); y++) {
      operator()(x, y) = other(x, y);
    }
  }
}
Matrix::Matrix() : m_width(0), m_height(0), m_ok(false) {}

Matrix::Matrix(const size_t &width, const size_t &height)
    : m_width(width), m_height(height), m_storage(width * height, 0.0) {}

Matrix::Matrix(const std::vector<std::vector<double>> &input) { init(input); }
void Matrix::init(const std::vector<std::vector<double>> &input) {
  m_height = input.size();
  if (input.size() > 0) {
    m_width = input[0].size();
    m_storage.resize(width() * height());
    for (size_t y = 0; y < input.size(); y++) {
      assert(input[y].size() == width());
      for (size_t x = 0; x < input[y].size(); x++) {
        operator()(x, y) = input[y][x];
      }
    }
  } else {
    m_width = 0;
    m_height = 0;
  }
}

Matrix Matrix::transposeROI() {
  return Matrix(MatrixROI(0, 0, width(), height(), this, 0, 0, true));
}

const Matrix Matrix::transposeROI() const {
  return Matrix(MatrixROI(0, 0, width(), height(), const_cast<Matrix *>(this),
                          0, 0, true));
}

Matrix Matrix::row(const size_t &v) {
  return Matrix(MatrixROI(0, v, this->width(), 1, this, 0, 0, false));
}
const Matrix Matrix::row(const size_t &v) const {
  return Matrix(MatrixROI(0, v, this->width(), 1, const_cast<Matrix *>(this), 0,
                          0, false));
}

Matrix Matrix::col(const size_t &u) {
  return Matrix(MatrixROI(u, 0, 1, this->height(), this, 0, 0, false));
}
const Matrix Matrix::col(const size_t &u) const {
  return Matrix(MatrixROI(u, 0, 1, this->height(), const_cast<Matrix *>(this),
                          0, 0, false));
}

std::ostream &operator<<(std::ostream &os, const Matrix &mat) {
  for (size_t y = 0; y < mat.height(); y++) {
    os << "[ ";
    for (size_t x = 0; x < mat.width(); x++) {
      os << std::to_string(mat(x, y)) << "\t";
    }
    os << " ]" << std::endl;
  }
  return os;
}
std::ostream &operator<<(std::ostream &os, const MatrixROI &roi) {
  os << "[ src_x=" << roi.srcX() << ",src_y=" << roi.srcY()
     << ",w=" << roi.width() << ",h=" << roi.height() << "dst_x=" << roi.dstX()
     << "dst_y=" << roi.dstY() << "trans=" << roi.transposed() << " ]";
  return os;
}

std::ostream &operator<<(std::ostream &os, const BoundingBox &bb) {
  os << "x=[ " << bb.minX() << "," << bb.maxX() << "), y=[" << bb.minY() << ","
     << bb.maxY() << ")";
  return os;
}

Matrix::Matrix(
    const std::initializer_list<std::initializer_list<double>> &input) {
  std::vector<std::vector<double>> input_vec;
  std::copy(input.begin(), input.end(), std::back_inserter(input_vec));
  init(input_vec);
}
double Matrix::det() const {
  // LUP determinant is much faster than brute force determinant.
  return lupDeterminant(*this);
}

Matrix::Matrix(const std::vector<double> &input) { init({input}); }
Matrix::Matrix(const std::initializer_list<double> &input) { init({input}); }

void Matrix::addROI(MatrixROI roi_to_add) {
  if (roi_to_add.width() == 0 || roi_to_add.height() == 0) {
    return;
  }
  for (auto roi : m_rois) {
    if (roi.overlaps(roi_to_add)) {
    }
  }
  m_rois.push_back(roi_to_add);
  size_t roi_x_bound = roi_to_add.dstX() + roi_to_add.dstWidth();
  this->m_width = std::max(this->width(), roi_x_bound);
  size_t roi_y_bound = roi_to_add.dstY() + roi_to_add.dstHeight();
  this->m_height = std::max(this->height(), roi_y_bound);
  m_ok = true;
}

Matrix::Matrix(MatrixROI roi) : m_width(0), m_height(0) { addROI(roi); }

Matrix::Matrix(std::vector<MatrixROI> &rois) : m_width(0), m_height(0) {
  for (auto roi : rois) {
    addROI(roi);
  }
}

double &Matrix::operator()(const size_t &x, const size_t &y) {
  for (auto roi : m_rois) {
    if (roi.isInside(x, y)) {
      size_t new_x = x;
      size_t new_y = y;
      roi.dstToSrc(x, y, new_x, new_y);
      return (*roi.matrix())(new_x, new_y);
    }
  }
  return m_storage[getIndex(x, y)];
}
size_t Matrix::width() const { return m_width; }
size_t Matrix::height() const { return m_height; }
bool Matrix::ok() const { return m_ok; }

Matrix identity(const size_t &size) {
  Matrix result(size, size);
  for (size_t i = 0; i < size; i++) {
    result(i, i) = 1.0;
  }
  return result;
}

size_t Matrix::getIndex(const size_t &x, const size_t &y) const {
  return width() * y + x;
}
double *Matrix::data() { return &m_storage[0]; }

bool Matrix::contiguous() const { return (m_rois.size() == 0); }

void Matrix::reshape(const size_t &new_width, const size_t &new_height) {
  if (new_width * new_height != this->width() * this->height()) {
    throw std::runtime_error(
        std::to_string(new_width) + " x" + std::to_string(new_height) +
        "new_w x new_h != w x h" + std::to_string(this->width()) + "," +
        std::to_string(this->height()) + "in reshape.");
  }
  if (m_rois.size() > 0) {
    throw std::runtime_error("Can't reshape a matrix mapping.");
  }
  this->m_width = new_width;
  this->m_height = new_height;
}

Matrix Matrix::sumRows() const {
  Matrix result(this->width(), 1);
  for (size_t i = 0; i < this->height(); i++) {
    result += this->row(i);
  }
  return result;
}

Matrix Matrix::sumCols() const {
  Matrix result(1, this->height());
  for (size_t i = 0; i < this->width(); i++) {
    result += this->col(i);
  }
  return result;
}

}; // namespace basic_matrix
