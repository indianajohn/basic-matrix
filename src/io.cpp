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

bool system_is_big_endian(void) {
  union {
    uint32_t i;
    char c[4];
  } bint = {0x01020304};

  return bint.c[0] == 1;
}

float read_bytes(std::ifstream &stream, bool switch_byte_order) {
  char bytes[4];
  stream.read(bytes, 4);
  if (switch_byte_order) {
    for (size_t i = 0; i < 2; i++) {
      char temp = bytes[i];
      bytes[i] = bytes[3 - i];
      bytes[3 - i] = temp;
    }
  }
  float val;
  memcpy(&val, &bytes, sizeof(bytes));
  return val;
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
  std::ofstream write_stream(path);
  write_stream << "Pf";
  write_stream.put(0x0a);
  write_stream << std::to_string(mat.width()) << " "
               << std::to_string(mat.height());
  write_stream.put(0x0a);
  write_stream << "-1.0";
  write_stream.put(0x0a);
  for (size_t i = 0; i < mat.height(); i++) {
    for (size_t j = 0; j < mat.width(); j++) {
      const float &val = mat(j, i);
      write_stream.write(reinterpret_cast<const char *>(&val), sizeof(float));
    }
  }
  return;
}

Matrix loadFromPfm(const std::string &str) {
  std::ifstream read_stream(str);
  std::string type, wh, byte_order;
  try {
    std::getline(read_stream, type);
    std::getline(read_stream, wh);
    std::getline(read_stream, byte_order);
  } catch (...) {
    throw std::runtime_error("Header parse error occurred when reading image " +
                             str);
  }
  if (type != "Pf") {
    throw std::runtime_error("File" + str + "was not of type Pf, instead was " +
                             type +
                             "; only type Pf (single-channel portable float "
                             "map) is supported by this reader.");
  }
  size_t delim = wh.find(" ");
  if (delim == std::string::npos) {
    throw std::runtime_error("dimensions in header of image " + str +
                             "could not be parsed: " + wh);
  }
  std::string w_string = wh.substr(0, delim);
  if (delim + 1 >= str.size()) {
    throw std::runtime_error("dimensions in header of image " + str +
                             "could not be parsed: " + wh);
  }
  std::string h_string = wh.substr(delim + 1);
  size_t h = 0;
  size_t w = 0;
  try {
    h = std::stoi(h_string);
    w = std::stoi(w_string);
  } catch (...) {
    throw std::runtime_error("dimensions in header of image " + str +
                             "could not be parsed: " + wh);
  }
  float byte_order_float = 1.0;
  try {
    byte_order_float = std::stof(byte_order);
  } catch (...) {
    throw std::runtime_error("byte order in header of image " + str +
                             "could not be parsed: " + byte_order);
  }
  bool file_is_big_endian = byte_order_float > 0;
  bool system_big_endian = system_is_big_endian();
  bool switch_byte_order = (file_is_big_endian && !system_big_endian) ||
                           (!file_is_big_endian && system_big_endian);
  Matrix loaded_matrix(w, h);
  for (size_t i = 0; i < h; i++) {
    for (size_t j = 0; j < w; j++) {
      loaded_matrix(j, i) = read_bytes(read_stream, switch_byte_order);
    }
  }
  return loaded_matrix;
}

}; // namespace basic_matrix
