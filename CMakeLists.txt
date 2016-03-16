cmake_minimum_required(VERSION 2.8)

OPTION(USE_LUAJIT "Link with luajit.so.  Wont work on Mac.  Faster on linux" ON)

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
  # force to not use luajit on Mac
  # why?  see http://stackoverflow.com/questions/20858911/lua-open-returns-null-using-luajit/20875342#20875342
  OPTION(USE_LUAJIT "Link with luajit.so.  Wont work on Mac.  Faster on linux" OFF)
endif()

FIND_PACKAGE(Torch REQUIRED)  # you need to run source ~/torch/install/bin/torch-activate if this line
                              # gives an error

SET(CMAKE_C_FLAGS "-std=c99 ")
SET(CMAKE_CXX_FLAGS "-std=c++0x -Wall")

include_directories(${TORCH_INSTALL}/include)
include_directories(${TORCH_INSTALL}/include/TH)

if(USE_LUAJIT)
  ADD_DEFINITIONS(-DUSE_LUAJIT)
else()
  ADD_DEFINITIONS(-DUSE_PYTORCHLUA)
endif()

add_library(PyTorchNative SHARED src/nnWrapper.cpp src/LuaHelper.cpp)
target_link_libraries(PyTorchNative luaT)
target_link_libraries(PyTorchNative TH)
set(INSTALL_TARGETS PyTorchNative)

if(USE_LUAJIT)
  target_link_libraries(PyTorchNative luajit)
else()
  ADD_DEFINITIONS(-DLUA_USE_LINUX)

  file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/lua-5.1.5/files.txt lua_src1)
  foreach(source ${lua_src1})
    set( lua_src ${lua_src} thirdparty/lua-5.1.5/src/${source})
  endforeach()

  add_library(PyTorchLua SHARED ${lua_src})
  target_link_libraries(PyTorchLua m)
  target_link_libraries(PyTorchNative PyTorchLua)
  set(INSTALL_TARGETS ${INSTALL_TARGETS} PyTorchLua)
endif()

install(TARGETS ${INSTALL_TARGETS}
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)

