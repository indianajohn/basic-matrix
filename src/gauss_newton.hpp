#pragma once
#include "optimization_problem.hpp"

namespace basic_matrix {
/// Use the Gauss-Newton method to solve an optimization
/// problem.
void gaussNewton(OptimizationProblem &problem);
};
