cmake_minimum_required(VERSION 3.7)
project(layer6)

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES main.c "inference6(hodai16).c" inference.c)
add_executable(layer6 ${SOURCE_FILES})