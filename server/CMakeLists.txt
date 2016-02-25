cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)
# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
${CUDA_NVCC_FLAGS};
    -x cu -O3 -gencode arch=compute_20,code=sm_20
    )
# Specify include directories
include_directories(
    src
    /usr/local/cuda/samples/common/inc
    )
# Specify library paths
link_directories()
# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    voxowl
    src/main.cu
    src/framebuffer.cu
    )
# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
    voxowl
    )
