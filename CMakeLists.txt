cmake_minimum_required(VERSION 2.8)

find_package(Lua51 REQUIRED)
include_directories(${LUA_INCLUDE_DIR})

set(HOME $ENV{HOME})
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/lua-5.1.5/src)
include_directories(${HOME}/torch/install/include/TH)
include_directories(${HOME}/torch/install/include)

#ADD_DEFINITIONS(-DLUA_USE_LINUX)

#file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/lua-5.1.5/files.txt lua_src1)
#foreach(source ${lua_src1})
#    set( lua_src ${lua_src} thirdparty/lua-5.1.5/src/${source})
#endforeach()

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-export-dynamic")
add_library(TorchLanguageIndependence SHARED nnWrapper.cpp LuaHelper.cpp)
#add_library(nnWrapper SHARED nnWrapper.cpp ${lua_src})
target_link_libraries(TorchLanguageIndependence ${LUA_LIBRARIES})
target_link_libraries(TorchLanguageIndependence ${HOME}/torch/install/lib/libluaT.so)
target_link_libraries(TorchLanguageIndependence ${HOME}/torch/install/lib/libTH.so)

install(TARGETS TorchLanguageIndependence
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)

