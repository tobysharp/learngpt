add_executable(affine_bench affine_bench.cpp)
target_compile_features(affine_bench PRIVATE cxx_std_23)
target_compile_options(affine_bench PRIVATE -g -fno-omit-frame-pointer)
target_include_directories(affine_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)

add_executable(thread_scaling_bench thread_scaling_bench.cpp)
target_compile_features(thread_scaling_bench PRIVATE cxx_std_23)
target_compile_options(thread_scaling_bench PRIVATE -g -fno-omit-frame-pointer)
target_include_directories(thread_scaling_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)

add_executable(vector_scaling_bench vector_scaling_bench.cpp)
target_compile_features(vector_scaling_bench PRIVATE cxx_std_23)
target_compile_options(vector_scaling_bench PRIVATE -g -fno-omit-frame-pointer)
target_include_directories(vector_scaling_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)