#ifndef VOXOWL_PLATFORM_H
#define VOXOWL_PLATFORM_H

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <cuda.h>
#include <driver_types.h>
#define GLM_FORCE_CUDA
#endif

#define GLM_SWIZZLE
#include <glm/glm.hpp>

#if defined(__CUDACC__)
#define VOXOWL_HOST_AND_DEVICE __host__ __device__
#define VOXOWL_DEVICE __device__
#define VOXOWL_HOST __host__
#define VOXOWL_CUDA_KERNEL __global__
#else
#define VOXOWL_HOST_AND_DEVICE 
#define VOXOWL_DEVICE 
#define VOXOWL_HOST
#define VOXOWL_CUDA_KERNEL
#endif

#if defined(HAVE_TURBOJPEG)
#define VOXOWL_USE_TURBOJPEG
#endif

#if defined(HAVE_CUDA)
#define VOXOWL_USE_CUDA
#endif

#define LIBDIVIDE_USE_SSE2 1

#endif
