cmake_minimum_required (VERSION 2.6)

project (zandbak)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/src/cmake)
find_package(SDL2)
#if(NOT SDL2_FOUND)
#  set(SDL2_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sdl-ubuntu12.04/include )
#  set(SDL2_LIBRARY SDL2)
#  link_directories( ${CMAKE_CURRENT_SOURCE_DIR}/sdl-ubuntu12.04/lib )
#endif()

#find libjpeg-turbo
#find_path(TJPEG_INCLUDE_DIR turbojpeg.h HINTS ${SDL2_INCLUDE_DIR})
#find_library(TJPEG_LIBRARY NAMES turbojpeg)

#include(FindPackageHandleStandardArgs)
#find_package_handle_standard_args(TJPEG DEFAULT_MSG TJPEG_LIBRARY TJPEG_INCLUDE_DIR)

#if(TJPEG_FOUND)
#    set(TJPEG_LIBRARIES ${TJPEG_LIBRARY})
#endif()

set (CMAKE_CXX_FLAGS "-std=c++11 -g")
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${SDL2_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIR})
#include_directories(${TJPEG_INCLUDE_DIR})
include_directories("../libvoxowl/include")

add_executable(zandbak src/main.cpp src/utils.cpp)

target_link_libraries (zandbak ${SDL2_LIBRARY})
