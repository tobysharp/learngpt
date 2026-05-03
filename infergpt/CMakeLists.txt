cmake_minimum_required(VERSION 3.28.3)

project(infergpt LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(
	$<$<CONFIG:Release>:-ffast-math>
	$<$<CONFIG:Release>:-mavx2>
	$<$<CONFIG:Release>:-mfma>
)

enable_testing()

add_subdirectory(src)
add_subdirectory(bench)
add_subdirectory(tests)