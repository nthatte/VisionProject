cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(iterative_closest_point)

find_package(PCL 1.6 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (icp src/icp.cpp)
target_link_libraries (icp ${PCL_LIBRARIES})
