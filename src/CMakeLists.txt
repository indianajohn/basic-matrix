
add_library(matrix matrix.cpp ops.cpp gaussian_elimination.cpp lup_decomposition.cpp inverse.cpp qr_factorization.cpp scalar.cpp gauss_newton.cpp optimization_problem.cpp k_means.cpp io.cpp standard_functions.cpp eigenvalues.cpp naive_gradient_descent.cpp knn.cpp)
target_include_directories(matrix PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(repl repl.cpp)
target_link_libraries(repl matrix)
