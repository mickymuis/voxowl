cmake_minimum_required(VERSION 2.8)

set( LOCAL_BUILD "~/local" )
set( NV_COMPUTE "52" )

find_package(CUDA REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -g -rdc=true -O3 -gencode arch=compute_${NV_COMPUTE},code=sm_${NV_COMPUTE}
    )

set( CUDA_SEPARABLE_COMPILATION ON )
set( CUDA_PROPAGATE_HOST_FLAGS OFF )
## Compiler flags
LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -g")

#SET( CUDA_HOST_COMPILER "g++44" )

#find libjpeg-turbo
find_path(TJPEG_INCLUDE_DIR turbojpeg.h HINTS ${LOCAL_BUILD}/include)
find_library(TJPEG_LIBRARY NAMES turbojpeg HINTS ${LOCAL_BUILD}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TJPEG DEFAULT_MSG TJPEG_LIBRARY TJPEG_INCLUDE_DIR)

if(TJPEG_FOUND)
        set(TJPEG_LIBRARIES ${TJPEG_LIBRARY})
        add_definitions(-DHAS_TURBOJPEG)
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
    src/mengersponge.cpp
    src/bmp.cpp
    src/raycast.cu
    src/raycast_cuda.cu
    ../libvoxowl/src/voxel.cu
    ../libvoxowl/src/svmipmap.cu
    ../libvoxowl/src/voxelmap.cpp
    )


# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
    voxowl
    stdc++
    ${TJPEG_LIBRARIES}
    )
