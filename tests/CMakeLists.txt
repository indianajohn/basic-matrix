
add_library(test_helpers test_helpers.cpp)
add_library(matrix_helpers matrix_helpers.cpp)
target_include_directories(test_helpers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(matrix_helpers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(matrix_helpers matrix)

function(prepare_matrix_test test_name source_file)
  add_executable(${test_name} ${source_file})
  target_link_libraries(${test_name} test_helpers matrix_helpers matrix)
  add_test(${test_name} ${test_name})
  set_target_properties (${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

function(untar relative_path_no_extension)
  add_custom_target(
    ${relative_path_no_extension}
    COMMAND tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/data/${relative_path_no_extension}.tar.gz
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/data/${relative_path_no_extension}.tar.gz
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
  )
endfunction()

# test data copy
configure_file(data/matrix.txt matrix.txt COPYONLY)
configure_file(data/1d_linear_regression.txt 1d_linear_regression.txt COPYONLY)
configure_file(data/2d_linear_regression.txt 2d_linear_regression.txt COPYONLY)
configure_file(data/logistic_regression_linear_decision_boundary.txt logistic_regression_linear_decision_boundary.txt COPYONLY)
configure_file(data/logistic_regression_circular_decision_boundary.txt logistic_regression_circular_decision_boundary.txt COPYONLY)

untar(knn_X_train.csv)
untar(knn_X_test.csv)
untar(knn_y_train.csv)
untar(knn_y_test.csv)

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
prepare_matrix_test(eigenvalues eigenvalues.cpp)
prepare_matrix_test(knn knn.cpp)
add_dependencies(knn knn_X_train.csv knn_X_test.csv knn_y_train.csv knn_y_test.csv)

