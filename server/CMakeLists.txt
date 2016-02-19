cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)
# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_22,code=sm_22
    )
# Specify include directories
include_directories(
    src
    )
# Specify library paths
link_directories()
# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    voxowl
    src/main.cpp
    src/framebuffer.cpp
    )
# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
    voxowl
    )
