cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -rdc=true -O3 -gencode arch=compute_20,code=sm_20
    )

set( CUDA_SEPARABLE_COMPILATION ON )

# Specify include directories
include_directories(
    src
    ../libvoxowl/include
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

## Compiler flags
LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -g -O3 -ffast-math -Wall")
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -rdc=true -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
    voxowl
    stdc++
    )
