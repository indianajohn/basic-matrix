#include "io.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace basic_matrix {
namespace {
std::vector<std::string> parseLine(const std::string &line) {
  std::vector<std::string> result;
  size_t next_comma = line.find(',');
  std::string line_copy = line;
  while (next_comma != std::string::npos) {
    std::string word = line_copy.substr(0, next_comma);
    result.push_back(word);
    line_copy = line_copy.substr(next_comma + 1);
    next_comma = line_copy.find(',');
  }
  if (line_copy.size() > 0) {
    result.push_back(line_copy);
  }
  return result;
}
std::vector<double> parseNumberLine(const std::string &line) {
  std::vector<std::string> words = parseLine(line);
  std::vector<double> result;
  for (const auto &word : words) {
    result.push_back(atof(word.c_str()));
  }
  return result;
}
}; // namespace

Matrix parseFromString(const std::string &str) {
  std::string cache = str;
  size_t next_newline = cache.find('\n');
  std::vector<std::vector<double>> matrix;
  size_t col_count = 0;
  while (next_newline != std::string::npos) {
    std::string line = cache.substr(0, next_newline);
    cache = cache.substr(next_newline + 1);
    auto number_line = parseNumberLine(line);
    if (col_count != 0 && number_line.size() != col_count) {
      throw std::runtime_error(
          "All rows in the string did not have the same element count.");
    }
    col_count = number_line.size();
    matrix.push_back(number_line);
    next_newline = cache.find('\n');
  }
  if (cache.size() > 0) {
    matrix.push_back(parseNumberLine(cache));
  }
  return Matrix(matrix);
}
Matrix loadFromFile(const std::string &str) {
  std::ifstream stream(str);
  std::string line;
  std::vector<std::vector<double>> matrix;
  while (std::getline(stream, line)) {
    matrix.push_back(parseNumberLine(line));
  }
  return Matrix(matrix);
}

std::string writeToString(const Matrix &mat) {
  std::stringstream result_stream;
  // 15 significant digits (double)
  result_stream << std::setw(15);
  for (size_t j = 0; j < mat.height(); j++) {
    for (size_t i = 0; i < mat.width(); i++) {
      result_stream << mat(i, j);
      if (i != mat.width() - 1) {
        result_stream << ",";
      }
    }
    if (j != mat.height() - 1) {
      result_stream << std::endl;
    }
  }
  return result_stream.str();
}

void writeToFile(const std::string &path, const Matrix &mat) {
  std::ofstream stream(path);
  // 15 significant digits (double)
  stream << std::setw(15);
  for (size_t j = 0; j < mat.height(); j++) {
    for (size_t i = 0; i < mat.width(); i++) {
      stream << mat(i, j);
      if (i != mat.width() - 1) {
        stream << ",";
      }
    }
    if (j != mat.height() - 1) {
      stream << std::endl;
    }
  }
}

void writeToPfm(const std::string &path, const Matrix &mat) {
  // TODO
  return;
}

Matrix loadFromPfm(const std::string &str) {
  Matrix loaded_matrix;
  // TODO
  return loaded_matrix;
}

}; // namespace basic_matrix
