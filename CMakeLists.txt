cmake_minimum_required(VERSION 2.8)

find_package(Lua51 REQUIRED)
include_directories(${LUA_INCLUDE_DIR})

set(HOME $ENV{HOME})
include_directories(${HOME}/torch/install/include/TH)

add_library(nnWrapper SHARED nnWrapper.cpp)
target_link_libraries(nnWrapper ${LUA_LIBRARY})

