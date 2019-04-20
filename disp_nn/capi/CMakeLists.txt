cmake_minimum_required(VERSION 3.10.1)

project(Vision VERSION 0.1)

find_package( OpenCV 4.0.1 REQUIRED )

set(BUILD_DIR "D:/projects/repo/TF_disp/disp_nn/capi/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "D:/projects/repo/TF_disp/disp_nn/capi/bin")
set(PROJECT_SOURCE_DIR "D:/projects/repo/TF_disp/disp_nn/capi/src")
set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)

include_directories(
    "D:/program/boost_1_69_0"
    "C:/Users/bsvse/AppData/Local/Programs/Python/Python37/include"
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(
    "D:/program/boost_1_69_0" 
    "D:/program/boost_1_69_0/stage/lib" 
    "C:/Users/bsvse/AppData/Local/Programs/Python/Python37/libs"
)

add_library(sgbm SHARED "${PROJECT_SOURCE_DIR}/sgbm.cpp")

target_link_libraries(sgbm ${OpenCV_LIBS})


