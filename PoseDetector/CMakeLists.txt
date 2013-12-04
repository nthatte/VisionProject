cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# we need FindFLANN.cmake 
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

project(vfh_pose_detector)

find_package(PCL 1.6 REQUIRED)
add_definitions(-DPCL17)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_library(VFHPoseEstimation src/VFHPoseEstimation.cpp)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeFiles/Modules/")

find_package(HDF5)
if(HDF5_FOUND)

    find_package(FLANN)
    include_directories(${FLANN_INCLUDE_DIRS})

    include_directories(${HDF5_INCLUDE_DIR})

    add_executable (vfhTrain src/vfhTrain.cpp)
    target_link_libraries (vfhTrain ${PCL_LIBRARIES} ${Boost_LIBRARIES}
        ${FLANN_LIBRARIES} ${HDF5_hdf5_LIBRARY} VFHPoseEstimator)

    add_executable (vfhTest src/vfhTest.cpp)
    target_link_libraries (vfhTest ${PCL_LIBRARIES} ${Boost_LIBRARIES}
        ${FLANN_LIBRARIES} ${HDF5_hdf5_LIBRARY} VFHPoseEstimator)

endif(HDF5_FOUND)
