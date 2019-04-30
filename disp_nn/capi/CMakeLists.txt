cmake_minimum_required(VERSION 3.10.1)

project(Vision VERSION 0.1)

set(BUILD_DIR "./build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../bin")
set(PROJECT_SOURCE_DIR "./src")
set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)

# need to change this according to the elas version
set (LIBELAS_SRC_DIR ${PROJECT_SOURCE_DIR}/elasCNNgrid)

# LIBELAS sources
FILE(GLOB LIBELAS_SRC_FILES "${LIBELAS_SRC_DIR}/*.cpp")

# use sse3 instruction set
set(CMAKE_CXX_FLAGS "-msse3")

include_directories(
    "${LIBELAS_SRC_DIR}"
    "F:/boost/boost_1_69_0"
    "C:/Python37/include"
)

link_directories(
    "F:/boost/boost_1_69_0" 
    "F:/boost/boost_1_69_0/stage/lib"
    "C:/Python37/libs"
)

#add_library(sgbm SHARED "${PROJECT_SOURCE_DIR}/sgbm.cpp")
#add_library(pyelas SHARED ${LIBELAS_SRC_FILES} "${PROJECT_SOURCE_DIR}/pyelas.cpp")
#add_library(elasCNN SHARED ${LIBELAS_SRC_FILES} "${PROJECT_SOURCE_DIR}/elasCNN.cpp")
#add_library(elasCNNstd SHARED ${LIBELAS_SRC_FILES} "${PROJECT_SOURCE_DIR}/elasCNNstd.cpp")
#add_library(elasCNNsup SHARED ${LIBELAS_SRC_FILES} "${PROJECT_SOURCE_DIR}/elasCNNsup.cpp")
add_library(elasCNNgrid SHARED ${LIBELAS_SRC_FILES} "${PROJECT_SOURCE_DIR}/elasCNNgrid.cpp")



