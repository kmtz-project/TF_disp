cmake_minimum_required(VERSION 3.6.2)

project(Vision VERSION 0.1)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../../")
set(PROJECT_SOURCE_DIR "../src")
set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)
set(LIST_LINK_LIBRARIES 
        boost_python37
        boost_numpy37
        python3.7m)

# ELAS Releases
set (PYELAS_SRC_DIR ${PROJECT_SOURCE_DIR}/elas)
FILE(GLOB PYELAS_SRC_FILES "${PYELAS_SRC_DIR}/*.cpp")

set (ELASCNN_SRC_DIR ${PROJECT_SOURCE_DIR}/elasCNN)
FILE(GLOB ELASCNN_SRC_FILES "${ELASCNN_SRC_DIR}/*.cpp")

set (ELASSTD_SRC_DIR ${PROJECT_SOURCE_DIR}/elasCNNstd)
FILE(GLOB ELASSTD_SRC_FILES "${ELAS_STD_SRC_DIR}/*.cpp")

set (ELASSUP_SRC_DIR ${PROJECT_SOURCE_DIR}/elasCNNsup)
FILE(GLOB ELASSUP_SRC_FILES "${ELASSUP_SRC_DIR}/*.cpp")

set (ELASGRID_SRC_DIR ${PROJECT_SOURCE_DIR}/elasCNNgrid)
FILE(GLOB ELASGRID_SRC_FILES "${ELASGRID_SRC_DIR}/*.cpp")


# use sse3 instruction set
set(CMAKE_CXX_FLAGS "-msse3")

include_directories(
    "/usr/local/include/python3.7"
)

link_directories(
    "/usr/local/lib/python3.7"
)

add_library(sgbm SHARED "${PROJECT_SOURCE_DIR}/sgbm.cpp")
add_library(pyelas SHARED ${PYELAS_SRC_FILES} "${PROJECT_SOURCE_DIR}/pyelas.cpp")
add_library(elasCNN SHARED ${ELASCNN_SRC_FILES} "${PROJECT_SOURCE_DIR}/elasCNN.cpp")
add_library(elasCNNstd SHARED ${ELASSTD_SRC_FILES} "${PROJECT_SOURCE_DIR}/elasCNNstd.cpp")
add_library(elasCNNsup SHARED ${ELASSUP_SRC_FILES} "${PROJECT_SOURCE_DIR}/elasCNNsup.cpp")
add_library(elasCNNgrid SHARED ${ELASGRID_SRC_FILES} "${PROJECT_SOURCE_DIR}/elasCNNgrid.cpp")

target_link_libraries(sgbm python3.7m boost_python37 boost_numpy37)
set_target_properties(sgbm PROPERTIES PREFIX "")

target_link_libraries(pyelas python3.7m boost_python37 boost_numpy37)
target_include_directories(pyelas PUBLIC ${PYELAS_SRC_DIR})
set_target_properties(pyelas PROPERTIES PREFIX "")

target_link_libraries(elasCNN python3.7m boost_python37 boost_numpy37)
target_include_directories(elasCNN PUBLIC ${ELASCNN_SRC_DIR})
set_target_properties(elasCNN PROPERTIES PREFIX "")

target_link_libraries(elasCNNstd python3.7m boost_python37 boost_numpy37)
target_include_directories(elasCNNstd PUBLIC ${ELASSTD_SRC_DIR})
set_target_properties(elasCNNstd PROPERTIES PREFIX "")

target_link_libraries(elasCNNsup python3.7m boost_python37 boost_numpy37)
target_include_directories(elasCNNsup PUBLIC ${ELASSUP_SRC_DIR})
set_target_properties(elasCNNsup PROPERTIES PREFIX "")

target_link_libraries(elasCNNgrid ${LIST_LINK_LIBRARIES})
target_include_directories(elasCNNgrid PUBLIC ${ELASGRID_SRC_DIR})
set_target_properties(elasCNNgrid PROPERTIES PREFIX "")


