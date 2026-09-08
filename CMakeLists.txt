cmake_minimum_required(VERSION 3.20)
project(e57cov CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The reader is plain C++20 with no dependencies so it builds and tests off the
# target platform. Only the forthcoming Metal layer will be macOS-only.
add_library(e57cov_core STATIC
  src/e57.cpp
  src/camera.cpp
  src/scan_check.cpp
  src/point_cloud.cpp
  src/picker.cpp
  src/frame.cpp
  src/lod.cpp
  src/point_store.cpp
  src/indexer.cpp
  src/range_image.cpp
  src/carve.cpp
  src/visibility.cpp)
target_include_directories(e57cov_core PUBLIC src)
target_compile_options(e57cov_core PRIVATE -Wall -Wextra)

add_executable(e57cov src/main.cpp)
target_link_libraries(e57cov PRIVATE e57cov_core)
target_compile_options(e57cov PRIVATE -Wall -Wextra)

enable_testing()
add_executable(test_e57 tests/test_e57.cpp)
target_link_libraries(test_e57 PRIVATE e57cov_core)
target_compile_options(test_e57 PRIVATE -Wall -Wextra)
add_test(NAME e57_reader COMMAND test_e57)

add_executable(test_viewer tests/test_viewer.cpp)
target_link_libraries(test_viewer PRIVATE e57cov_core)
target_compile_options(test_viewer PRIVATE -Wall -Wextra)
add_test(NAME viewer COMMAND test_viewer)

add_executable(test_lod tests/test_lod.cpp)
target_link_libraries(test_lod PRIVATE e57cov_core)
target_compile_options(test_lod PRIVATE -Wall -Wextra)
add_test(NAME lod COMMAND test_lod)

add_executable(test_indexer tests/test_indexer.cpp)
target_link_libraries(test_indexer PRIVATE e57cov_core)
target_compile_options(test_indexer PRIVATE -Wall -Wextra)
add_test(NAME indexer COMMAND test_indexer)

add_executable(test_range_image tests/test_range_image.cpp)
target_link_libraries(test_range_image PRIVATE e57cov_core)
target_compile_options(test_range_image PRIVATE -Wall -Wextra)
add_test(NAME range_image COMMAND test_range_image)

add_executable(test_carve tests/test_carve.cpp)
target_link_libraries(test_carve PRIVATE e57cov_core)
target_compile_options(test_carve PRIVATE -Wall -Wextra)
add_test(NAME carve COMMAND test_carve)

add_executable(test_visibility tests/test_visibility.cpp)
target_link_libraries(test_visibility PRIVATE e57cov_core)
target_compile_options(test_visibility PRIVATE -Wall -Wextra)
add_test(NAME visibility COMMAND test_visibility)
