cmake_minimum_required(VERSION 3.28.3)

project(infergpt LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(infergpt main.cpp)
target_compile_features(infergpt PRIVATE cxx_std_23)