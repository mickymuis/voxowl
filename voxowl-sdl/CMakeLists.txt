cmake_minimum_required (VERSION 2.6)

project (voxowl-sdl)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/src/cmake)
find_package(SDL2)
if(NOT SDL2_FOUND)
  set(SDL2_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sdl-ubuntu12.04/include )
  set(SDL2_LIBRARY SDL2)
  link_directories( ${CMAKE_CURRENT_SOURCE_DIR}/sdl-ubuntu12.04/lib )
endif()
set (CMAKE_CXX_FLAGS "-std=c++0x")
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${SDL2_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIR})
include_directories("../libvoxowl/include")

add_executable(voxowl-sdl src/main.c src/voxowl_network.c)

target_link_libraries (voxowl-sdl ${SDL2_LIBRARY})
