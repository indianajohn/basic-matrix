#include <ctype.h>
#include <iostream>
#include <matrix.hpp>
#include <optional>
#include <string>

using namespace basic_matrix;

/*
 * Parse user input and decide when to end input.
 */
std::string readCommandFromConsole() {
  std::cout << "# ";
  std::string cmd = "";
  std::getline(std::cin, cmd);
  return cmd;
}

std::string strip(const std::string &expr) {
  std::string out;
  for (const auto &c : expr) {
    if (c != ' ') {
      out.push_back(c);
    }
  }
  return out;
}

std::vector<std::string> split(const std::string &expr, const char &delim) {
  std::vector<std::string> tokens;
  int last_i = -1;
  int i = expr.find(delim, 0);
  while (i != std::string::npos && i + 1 < expr.size() && last_i < i) {
    std::string token = expr.substr(last_i + 1, i - last_i - 1);
    if (token.size() > 0) {
      tokens.push_back(token);
    }
    last_i = i;
    i = expr.find(delim, i + 1);
  }
  if (last_i > 0) {
    std::string token = expr.substr(last_i + 1);
    if (token.size() > 0) {
      tokens.push_back(token);
    }
  }
  if (tokens.size() == 0 && expr.size() > 0) {
    tokens.push_back(expr);
  }
  return tokens;
}

void printTokens(const std::vector<std::string> tokens) {
  for (const auto &token : tokens) {
    std::cout << "\"" << token << "\",";
  }
  std::cout << std::endl;
}

bool enclosedIn(const char &left, const char &right, const std::string &expr) {
  if (expr.size() < 3) {
    return false;
  }
  return expr[0] == left && expr[expr.size() - 1] == right;
}

class Parser {
public:
  void parseExpression(std::string expr) {
    expr = strip(expr);
    // Try to parse an equals expression.
    std::vector<std::string> tokens = split(expr, '=');
    if (tokens.size() == 2) {
      equals(tokens[0], tokens[1]);
    } else if (tokens.size() > 2) {
      std::cout << "Invalid equals statement:" << expr << std::endl;
    }
    // Not equals; try to print a variable.
    const auto result = this->parseMatrixExpression(expr);
    if (result) {
      std::cout << *result << std::endl;
    }
  }

  std::optional<Matrix> parseMatrix(std::string expr) {
    if (expr.size() < 3) {
      return std::optional<Matrix>();
    }
    expr = expr.substr(1, expr.size() - 2);
    std::vector<std::string> row_strings = split(expr, ';');
    std::vector<std::vector<std::string>> rows;
    for (const auto &row_string : row_strings) {
      std::vector<std::string> row = split(row_string, ',');
      if (rows.size() != 0 && rows[0].size() != row.size()) {
        return std::optional<Matrix>();
      }
      rows.push_back(row);
    }
    std::vector<std::vector<double>> raw_matrix;
    for (const auto &row : rows) {
      raw_matrix.resize(raw_matrix.size() + 1);
      for (const auto &col : row) {
        raw_matrix.back().push_back(atof(col.c_str()));
      }
    }
    return Matrix(raw_matrix);
  }

  std::optional<Matrix>
  evaluateOperator(const std::string &expr, const char &op,
                   const std::function<Matrix(Matrix, Matrix)> &fn) {
    std::vector<std::string> tokens = split(expr, op);
    if (tokens.size() > 1) {
      auto base_matrix = parseMatrixExpression(tokens[0]);
      if (base_matrix) {
        for (size_t i = 1; i < tokens.size(); i++) {
          auto matrix = parseMatrixExpression(tokens[i]);
          if (matrix) {
            try {
              *base_matrix = fn(*base_matrix, *matrix);
            } catch (const std::exception &e) {
              std::cout << "Operator " << op
                        << " failed with exception: " << std::endl
                        << e.what() << std::endl;
              std::cout << "When evaluating token " << i << " of:" << std::endl;
              printTokens(tokens);
              return std::optional<Matrix>();
            }
          }
        }
        return base_matrix;
      }
    }
    return std::optional<Matrix>();
  }
  std::optional<Matrix>
  evaluateUnaryOperator(const std::string &expr, const char &op,
                        const std::function<Matrix(Matrix)> &fn) {
    std::vector<std::string> tokens = split(expr, '~');
    if (tokens.size() == 2) {
      if (tokens[1].size() == 1 && tokens[1][0] == op) {
        auto mat = parseMatrixExpression(tokens[0]);
        if (mat) {
          try {
            return fn(*mat);
          } catch (const std::exception &e) {
            std::cout << "Expression " << expr
                      << " failed with exception:" << std::endl
                      << e.what() << std::endl;
          }
        }
      }
    }
    return std::optional<Matrix>();
  }

  std::optional<Matrix> parseMatrixExpression(const std::string &expr) {
    Matrix matrix;
    if (enclosedIn('[', ']', expr)) {
      return parseMatrix(expr);
    } else if (validateVariableName(expr) &&
               m_workspace.find(expr) != m_workspace.end()) {
      return m_workspace.at(expr);
    } else if (enclosedIn('(', ')', expr) && expr.size() > 2) {
      return this->parseMatrixExpression(expr.substr(1, expr.size() - 2));
    }
    auto result = evaluateUnaryOperator(
        expr, 'T', [](const auto &m) { return m.transpose(); });
    if (result)
      return result;
    result = evaluateUnaryOperator(expr, 'I',
                                   [](const auto &m) { return m.inverse(); });
    if (result)
      return result;
    result = evaluateOperator(
        expr, '+', [](const auto &a, const auto &b) { return a + b; });
    if (result)
      return result;
    result = evaluateOperator(
        expr, '-', [](const auto &a, const auto &b) { return a - b; });
    if (result)
      return result;
    result = evaluateOperator(
        expr, '*', [](const auto &a, const auto &b) { return a * b; });
    if (result)
      return result;
    return std::optional<Matrix>();
  }

private:
  bool validateVariableName(const std::string &name) {
    for (const auto &c : name) {
      if (!isalpha(c) && c != '_') {
        return false;
      }
    }
    return true;
  }
  void equals(const std::string &lhs, const std::string &rhs) {
    if (this->validateVariableName(lhs)) {
      const auto parsed = parseMatrixExpression(rhs);
      if (parsed) {
        m_workspace[lhs] = *parsed;
      } else {
        std::cout << "Could not parse " << rhs << " as a matrix expression."
                  << std::endl;
      }
    } else {
      std::cout << lhs << " is not a valid variable name." << std::endl;
    }
  }
  std::unordered_map<std::string, basic_matrix::Matrix> m_workspace;
};

int main(int argc, char **argv) {
  std::string cmd = "";
  Parser parser;
  while (cmd != "exit") {
    cmd = readCommandFromConsole();
    parser.parseExpression(cmd);
  }
}
