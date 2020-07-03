
add_library(test_helpers test_helpers.cpp)
add_library(matrix_helpers matrix_helpers.cpp)
target_include_directories(test_helpers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(matrix_helpers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(matrix_helpers matrix)

function(prepare_matrix_test test_name source_file)
  add_executable(${test_name} ${source_file})
  target_link_libraries(${test_name} test_helpers matrix_helpers matrix)
  add_test(${test_name} ${test_name})
endfunction()

# test data copy
configure_file(data/matrix.txt matrix.txt COPYONLY)
configure_file(data/1d_linear_regression.txt 1d_linear_regression.txt COPYONLY)
configure_file(data/2d_linear_regression.txt 2d_linear_regression.txt COPYONLY)

prepare_matrix_test(basic matrix.cpp)
prepare_matrix_test(ops ops.cpp)
prepare_matrix_test(gaussian_elimination gaussian_elimination.cpp)
prepare_matrix_test(lup_decomposition lup_decomposition.cpp)
prepare_matrix_test(inverse inverse.cpp)
prepare_matrix_test(qr_factorization qr_factorization.cpp)
prepare_matrix_test(optimization_problem optimization_problem.cpp)
prepare_matrix_test(gauss_newton gauss_newton.cpp)
prepare_matrix_test(k_means k_means.cpp)
prepare_matrix_test(io io.cpp)
prepare_matrix_test(standard_functions standard_functions.cpp)

