
add_library(matrix matrix.cpp ops.cpp gaussian_elimination.cpp lup_decomposition.cpp inverse.cpp qr_factorization.cpp scalar.cpp gauss_newton.cpp optimization_problem.cpp k_means.cpp io.cpp)
target_include_directories(matrix PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
