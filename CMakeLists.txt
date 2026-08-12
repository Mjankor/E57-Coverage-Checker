cmake_minimum_required(VERSION 3.20)
project(e57cov CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(e57cov_core STATIC src/e57.cpp)
target_include_directories(e57cov_core PUBLIC src)
target_compile_options(e57cov_core PRIVATE -Wall -Wextra)

enable_testing()
add_executable(test_e57 tests/test_e57.cpp)
target_link_libraries(test_e57 PRIVATE e57cov_core)
target_compile_options(test_e57 PRIVATE -Wall -Wextra)
add_test(NAME e57_reader COMMAND test_e57)
