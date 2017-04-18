#ifndef LIBSVMM_PLATFORM_H
#define LIBSVMM_PLATFORM_H

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <cuda.h>
#include <driver_types.h>
#define GLM_FORCE_CUDA
#endif

#define GLM_SWIZZLE
//#include <glm/glm.hpp>

#if defined(__CUDACC__)
#ifndef SVMM_HOST_AND_DEVICE
#define SVMM_HOST_AND_DEVICE __host__ __device__
#endif
#ifndef SVMM_DEVICE
#define SVMM_DEVICE __device__
#endif
#ifndef SVMM_HOST
#define SVMM_HOST __host__
#endif
#ifndef SVMM_CUDA_KERNEL
#define SVMM_CUDA_KERNEL __global__
#endif
#else
#ifndef SVMM_HOST_AND_DEVICE
#define SVMM_HOST_AND_DEVICE
#endif
#ifndef SVMM_DEVICE
#define SVMM_DEVICE
#endif
#ifndef SVMM_HOST
#define SVMM_HOST 
#endif
#ifndef SVMM_CUDA_KERNEL
#define SVMM_CUDA_KERNEL
#endif
#endif

#define LIBDIVIDE_USE_SSE2 1

#endif
