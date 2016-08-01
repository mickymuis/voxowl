#pragma once

#include "raycast.h"
#include "svmipmap.h"

typedef struct {
    voxel_format_t      format;
    int                 mipmap_factor;
    int                 block_count;
    int                 blockwidth;
    int                 texels_per_blockwidth;
    glm::ivec3          grid_size;
    int                 grid_y_shift;
    int                 grid_z_shift;
    cudaArray*          data_ptr;
    cudaTextureObject_t texture;
} svmm_level_t;

typedef svmm_level_t* svmm_level_devptr_t;

typedef struct {
    glm::ivec3          size;
/*    voxel_format_t      root_format;
    cudaArray*          root_data_ptr;
    cudaTextureObject_t root_texture;
    glm::ivec3          root_size;*/
    int                 levels;
    svmm_level_devptr_t level_ptr;
} svmipmapDevice_t;

VOXOWL_DEVICE fragment_t svmmRaycast( svmipmapDevice_t *v, const ray_t& r ); 

VOXOWL_HOST cudaError_t svmmCopyToDevice( svmipmapDevice_t* d_svmm, svmipmap_t* svmm );

