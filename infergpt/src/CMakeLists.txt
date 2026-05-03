cmake_minimum_required(VERSION 3.28.3)

project(infergpt LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

add_executable(infergpt main.cpp)
target_compile_features(infergpt PRIVATE cxx_std_23)

function(add_infergpt_test target)
	add_executable(${target} ../tests/${target}.cpp)
	target_compile_features(${target} PRIVATE cxx_std_23)
	target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
	add_test(NAME ${target} COMMAND ${target})
endfunction()

add_infergpt_test(matrix_test)
add_infergpt_test(algebra_test)