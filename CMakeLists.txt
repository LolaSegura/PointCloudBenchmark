
cmake_minimum_required(VERSION 3.10)
project(PointCloudBenchmark)

set(CMAKE_CXX_STANDARD 14)

# Find PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Find Open3D
find_package(Open3D REQUIRED)

# Add executable
add_executable(point_cloud_benchmark point_cloud_benchmark.cpp)
target_link_libraries(point_cloud_benchmark ${PCL_LIBRARIES} ${Open3D_LIBRARIES})