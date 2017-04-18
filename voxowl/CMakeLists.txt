
# This file is part of the Voxowl volumetric rendering toolkit
#
# See also: https://github.com/mickymuis/voxowl
#

cmake_minimum_required(VERSION 2.8)
project(voxowl)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set( LOCAL_BUILD "~/local" )
set( NV_COMPUTE "52" )
set( NV_SM "35" )
list(APPEND SOURCES    
    src/main.cpp
    src/testing.cpp
    src/core/actor.cpp
    src/core/object.cpp
    src/core/parser.cpp
    src/core/application.cpp
    src/network/packetbuffer.cpp
    src/network/network.cpp
    src/network/server.cpp
    src/render/rendertarget.cpp
    src/render/glrendertarget.cpp
    src/render/framebuffer.cpp
    src/render/fbrendertarget.cpp
    src/render/renderer.cpp
    src/render/volume_loader.cpp
    src/render/camera.cpp
    src/util/mengersponge.cpp
    src/util/bmp.cpp
    src/util/jpegencoder.cpp
    src/util/performance_counter.cpp
    src/svmm/voxelmap.cpp
    src/module/server_module.cpp)
list(APPEND SOURCES_CUDA
    src/cuda/raycast.cu
    src/cuda/raycast_cuda.cu
    src/cuda/dda_cuda.cu
    src/cuda/svmm_cuda.cu
    src/cuda/ivec3_32.cu
    src/cuda/voxel.cu
    src/cuda/svmipmap.cu)
list(APPEND SOURCES_NO_CUDA
    src/svmm/ivec3_32.cpp
    src/svmm/voxel.cpp
    src/svmm/svmipmap.cpp)

## Compiler flags
LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -g")

## Find CUDA, it is now optional
find_package(CUDA)
if(CUDA_FOUND)
    add_definitions(-DHAVE_CUDA)
    # Pass options to NVCC
    set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -g -rdc=true -O3 -std=c++11 --maxrregcount 32 --use_fast_math -lineinfo -gencode arch=compute_${NV_COMPUTE},code=compute_${NV_COMPUTE}
        )

    set( CUDA_SEPARABLE_COMPILATION ON )
    set( CUDA_PROPAGATE_HOST_FLAGS OFF )
endif()

## Find SDL2
find_package(SDL2)
if(NOT SDL2_FOUND)
    message(WARNING "SDL2 was NOT found, Voxowl will still build but the GUI or output to screen will NOT be available.")
#set(SDL2_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sdl-ubuntu12.04/include )
#set(SDL2_LIBRARY SDL2)
#link_directories( ${CMAKE_CURRENT_SOURCE_DIR}/sdl-ubuntu12.04/lib )
else()
    add_definitions(-DHAVE_SDL)
endif()

## Find libjpeg-turbo
find_path(TJPEG_INCLUDE_DIR turbojpeg.h HINTS ${LOCAL_BUILD}/include)
find_library(TJPEG_LIBRARY NAMES turbojpeg HINTS ${LOCAL_BUILD}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TJPEG DEFAULT_MSG TJPEG_LIBRARY TJPEG_INCLUDE_DIR)

if(TJPEG_FOUND)
        set(TJPEG_LIBRARIES ${TJPEG_LIBRARY})
        add_definitions(-DHAVE_TURBOJPEG)
endif()

## Find OpenMP
include (FindOpenMP)  
if (OPENMP_FOUND)
  set (HAVE_OPENMP TRUE)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
else()
  set ( HAVE_OPENMP FALSE)
endif()

## Find OpenGL
find_package(OpenGL)
if (OPENGL_FOUND)
    add_definitions(-DHAVE_OPENGL)
endif()

# Specify include directories
include_directories(
    src
    ../libsvmm/include
    ../ext/glm
    ${TJPEG_INCLUDE_DIR}
    ${SDL2_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR} )

# Specify library paths
#link_directories()

# Depending on whether we have CUDA or not, we compile using NVCC
if(CUDA_FOUND)
    cuda_add_executable(voxowl ${SOURCES} ${SOURCES_CUDA})
else()
    add_executable(voxowl ${SOURCES} ${SOURCES_NO_CUDA})
endif()

# Specify target & libraries to link it with
target_link_libraries(
    voxowl
    stdc++
    ${TJPEG_LIBRARIES}
    ${SDL2_LIBRARY}
    ${OPENGL_LIBRARIES} )

