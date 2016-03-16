cmake_minimum_required(VERSION 2.8)

FIND_PACKAGE(Torch REQUIRED)  # you need to run source ~/torch/install/bin/torch-activate if this line
                              # gives an error

SET(CMAKE_C_FLAGS "-std=c99 ")
SET(CMAKE_CXX_FLAGS "-std=c++0x -Wall")

include_directories(${TORCH_INSTALL}/include)
include_directories(${TORCH_INSTALL}/include/TH)

add_library(PyTorchNative SHARED src/nnWrapper.cpp src/LuaHelper.cpp)
target_link_libraries(PyTorchNative luaT)
target_link_libraries(PyTorchNative TH)
target_link_libraries(PyTorchNative luajit)

install(TARGETS PyTorchNative
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)

