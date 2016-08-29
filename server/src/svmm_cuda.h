#pragma once

#include "raycast.h"
#include "svmipmap.h"

typedef struct {
    voxel_format_t      format;
    int                 mipmap_factor;
    int                 mipmap_factor_log2;
    int                 block_count;
    int                 blockwidth;
    int                 blockwidth_log2; 
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

typedef struct {
    int levels;
    svmm_level_t *level_ptr;
    svmm_level_devptr_t level_devptr;
} svmipmapHost_t;

VOXOWL_DEVICE void svmmRaycast( fragment_t&, svmipmapDevice_t *v, box_t&, ray_t&, glm::vec3&,
        /* glm::vec3&,*/ float &fragment_width, const float& fragment_width_step ); 

VOXOWL_HOST cudaError_t svmmCopyToDevice( svmipmapDevice_t* d_svmm, svmipmap_t* svmm, svmipmapHost_t* svmm_host );
VOXOWL_HOST cudaError_t svmmFreeDevice( svmipmapHost_t* svmm_host );

