cmake_minimum_required(VERSION 3.10.1)

project(Vision VERSION 0.1)

set(BUILD_DIR "./build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../bin")
set(PROJECT_SOURCE_DIR "./src")
set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)

set (LIBELAS_SRC_DIR ${PROJECT_SOURCE_DIR}/elas)

# LIBELAS sources
FILE(GLOB LIBELAS_SRC_FILES "${LIBELAS_SRC_DIR}/*.cpp")

# use sse3 instruction set
set(CMAKE_CXX_FLAGS "-msse3")

include_directories(
    "${LIBELAS_SRC_DIR}"
    "D:/program/boost_1_69_0"
    "C:/Users/bsvse/AppData/Local/Programs/Python/Python37/include"
)

link_directories(
    "D:/program/boost_1_69_0" 
    "D:/program/boost_1_69_0/stage/lib" 
    "C:/Users/bsvse/AppData/Local/Programs/Python/Python37/libs"
)

add_library(sgbm SHARED "${PROJECT_SOURCE_DIR}/sgbm.cpp")
add_library(pyelas SHARED ${LIBELAS_SRC_FILES} "${PROJECT_SOURCE_DIR}/pyelas.cpp")


