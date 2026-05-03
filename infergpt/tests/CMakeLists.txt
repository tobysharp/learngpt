function(add_infergpt_test target)
  add_executable(${target} ${target}.cpp)
  target_compile_features(${target} PRIVATE cxx_std_23)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
  add_test(NAME ${target} COMMAND ${target})
endfunction()

add_infergpt_test(matrix_test)
add_infergpt_test(algebra_test)