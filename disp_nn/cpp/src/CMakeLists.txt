cmake_minimum_required(VERSION 3.6)

project(tftest)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

include_directories(
    /usr/local/include/tensorflow 
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(tftest main.cpp)

link_directories(/home/bsv/work/tensorflow-1.13.1/bazel-bin/tensorflow)
target_link_libraries(tftest tensorflow tensorflow_framework ${OpenCV_LIBS} pthread)
