cmake_minimum_required(VERSION 2.8)

set( LOCAL_BUILD "~/local" )

find_package(CUDA REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -rdc=true -O3 -gencode arch=compute_20,code=sm_20
    )

set( CUDA_SEPARABLE_COMPILATION ON )
set( CUDA_PROPAGATE_HOST_FLAGS OFF )
## Compiler flags
LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -g")
#LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -g -O3 -ffast-math -Wall")
#LIST(APPEND CUDA_NVCC_FLAGS -lineinfo -use_fast_math -Xptxas -dlcm=cg)

SET( CUDA_HOST_COMPILER "g++44" )

#find libjpeg-turbo
find_path(TJPEG_INCLUDE_DIR turbojpeg.h HINTS ${LOCAL_BUILD}/include)
find_library(TJPEG_LIBRARY NAMES turbojpeg)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TJPEG DEFAULT_MSG TJPEG_LIBRARY TJPEG_INCLUDE_DIR)

if(TJPEG_FOUND)
        set(TJPEG_LIBRARIES ${TJPEG_LIBRARY})
endif()

# Specify include directories
include_directories(
    src
    ../libvoxowl/include
    ${TJPEG_INCLUDE_DIR}
    )
# Specify library paths
link_directories()
# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    voxowl
    src/main.cpp
    src/object.cpp
    src/packetbuffer.cpp
    src/network.cpp
    src/server.cpp
    src/parser.cpp
    src/framebuffer.cpp
    src/renderer.cpp
    src/volume.cpp
    src/camera.cpp
    src/actor.cpp
    src/voxel.cu
    src/mengersponge.cpp
    src/bmp.cpp
    src/raycast.cu
    src/raycast_cuda.cu
    )


# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
    voxowl
    stdc++
    ${TJPEG_LIBRARIES}
    )
