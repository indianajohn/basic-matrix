#pragma once
#include "optimization_problem.hpp"

namespace basic_matrix {
/// Use naive gradient descent to solve the problem. Only supports gradient/
/// cost function. Uses a line-search method.
void naiveGradientDescent(OptimizationProblem &problem, const double &alpha);
}; // namespace basic_matrix
