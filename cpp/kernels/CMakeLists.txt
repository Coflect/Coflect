cmake_minimum_required(VERSION 3.18)
project(coflect_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(coflect_kernels STATIC kernel_stub.cpp)
target_compile_features(coflect_kernels PRIVATE cxx_std_17)
