#include <cuda_runtime.h>
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#define VOXOWL_HOST_AND_DEVICE __host__ __device__
#define VOXOWL_DEVICE __device__
#define VOXOWL_HOST __host__
#define VOXOWL_CUDA_KERNEL __global__

